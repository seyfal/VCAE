# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from skimage.filters import gaussian

from scipy.ndimage import uniform_filter

import hyperspy.api as hs

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import os
import sys 
import numpy as np
from datetime import datetime
from itertools import product
from operator import itemgetter
import json
from threading import Lock
import traceback
import optuna

class CVAE3D(nn.Module):
    """Convolutional variational autoencoder."""
    
    def __init__(self, latent_dim, size):
        super(CVAE3D, self).__init__()
        self.latent_dim = latent_dim
        self.size = size
        reduced_size = size // 8

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * reduced_size * reduced_size * reduced_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * reduced_size * reduced_size * reduced_size),
            nn.ReLU(),
            nn.Unflatten(1, (128, reduced_size, reduced_size, reduced_size)),
            nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = torch.sigmoid(logits)
            return probs
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

############################################################################################################
##################################### END OF MODEL DEFINITION ##############################################
############################################################################################################

def load_dm4_data(filepath):
    s = hs.load(filepath)
    data = s.data  # The 3D data array
    return data

def preprocess_3d_images(image, size, sigma, energy_range, xy_window):
    # Apply Gaussian blur directly to the single image
    blurred_image = gaussian(image, sigma=sigma, mode='reflect', preserve_range=True)
    
    # Calculate the pixel indices corresponding to the energy range
    start_pixel = int((energy_range[0] - 0))
    end_pixel = int((energy_range[1] - 0))
    
    # Slice the data array to keep only the desired energy range in the third dimension
    blurred_image = blurred_image[:, :, start_pixel:end_pixel]
    
    # Apply min-max scaling
    min_val = np.min(blurred_image)
    max_val = np.max(blurred_image)
    normalized_image = (blurred_image - min_val) / (max_val - min_val)
    
    # Apply spatial-spectral smoothing
    def smooth_spatial_spectral(arr, window):
        # Use uniform_filter to compute the sum of spectra in the neighborhood
        neighborhood_sum = uniform_filter(arr, size=(window, window, 1), mode='reflect')
        # Compute the number of pixels in the neighborhood
        neighborhood_count = uniform_filter(np.ones_like(arr), size=(window, window, 1), mode='reflect')
        # Compute the average
        return neighborhood_sum / neighborhood_count
    
    smoothed_img = smooth_spatial_spectral(normalized_image, xy_window)
    
    # Calculate the padding for each dimension
    padding = [(max(0, size - dim_size) // 2, max(0, size - dim_size) - max(0, size - dim_size) // 2) 
               for dim_size in smoothed_img.shape]
    
    # Apply padding
    padded_img = np.pad(smoothed_img, padding, mode='constant')
    
    # Calculate the crop for each dimension
    crop = [(max(0, padded_img.shape[i] - size) // 2, 
             max(0, padded_img.shape[i] - size) - max(0, padded_img.shape[i] - size) // 2) 
            for i in range(len(padded_img.shape))]
    
    # Apply cropping
    cropped_img = padded_img[crop[0][0]:padded_img.shape[0]-crop[0][1],
                             crop[1][0]:padded_img.shape[1]-crop[1][1],
                             crop[2][0]:padded_img.shape[2]-crop[2][1]]
    
    # Ensure the final shape matches the target size
    assert cropped_img.shape == (size, size, size), f"Shape mismatch: {cropped_img.shape} != {(size, size, size)}"
    
    # Reshape to (1, size, size, size) for PyTorch
    reshaped_image = cropped_img.reshape((1, 1, size, size, size))
    
    return cropped_img, reshaped_image.astype('float32')

############################################################################################################
##################################### END OF DATA PREPROCESSING ############################################
############################################################################################################
    
def compute_loss(model, x, kl_weight=0.5):
    # Check if the model is wrapped in DistributedDataParallel
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    
    reconstruction_loss = F.mse_loss(x_logit, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    total_loss = reconstruction_loss + kl_weight * kl_loss
    
    return total_loss, reconstruction_loss, kl_loss

def log_normal_pdf(sample, mean, logvar):
    log2pi = torch.log(torch.tensor(2. * np.pi))
    return torch.sum(-0.5 * ((sample - mean) ** 2 * torch.exp(-logvar) + logvar + log2pi), dim=1)

def gaussian_blur(img, sigma):
    return np.array(gaussian(img, (sigma, sigma)))

def gaussian_blur_arr(images, sigma):
    return np.array([gaussian_blur(img, sigma) for img in images])

def norm_max_pixel(images):
    return np.array([img / np.max(img) for img in images])

def visualize_inference(model, actual_image, cropped_img, energy_range=(900, 1100), pixel_x=None, pixel_y=None):
    """
    Visualize the actual image, preprocessed input, and model prediction for EELS data.
    
    Args:
    model: The trained CVAE3D model
    actual_image (numpy.ndarray): The original 3D EELS image
    cropped_img (numpy.ndarray): The preprocessed 3D image used as input to the model
    energy_range (tuple): The energy range (start, end) in eV
    pixel_x (int, optional): X coordinate of the pixel to show spectrum. If None, the center pixel is used.
    pixel_y (int, optional): Y coordinate of the pixel to show spectrum. If None, the center pixel is used.

    Returns:
    None (displays the plot)
    """
    model.eval()
    device = next(model.parameters()).device

    # If pixel coordinates are not provided, use the center pixel
    if pixel_x is None:
        pixel_x = cropped_img.shape[0] // 2
    if pixel_y is None:
        pixel_y = cropped_img.shape[1] // 2

    with torch.no_grad():
        # Prepare input tensor
        input_tensor = torch.tensor(cropped_img).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # Get model prediction
        mean, logvar = model.encode(input_tensor)
        z = model.reparameterize(mean, logvar)
        prediction = model.decode(z)
        
        # Move prediction to CPU and convert to numpy
        prediction_np = prediction.squeeze().cpu().numpy()

    # Create a figure with 3 rows and 2 columns
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    
    # Calculate the pixel indices corresponding to the energy range
    start_pixel = int((energy_range[0] - 0))
    end_pixel = int((energy_range[1] - 0))

    # Slice the data array to keep only the desired energy range in the third dimension
    actual_image = actual_image[:, :, start_pixel:end_pixel]

    # Plot 2D spatial images (sum along energy axis)
    images = [actual_image, cropped_img, prediction_np]
    titles = ['Actual Image', 'Preprocessed Input', 'Model Prediction']

    for i, (img, title) in enumerate(zip(images, titles)):
        spatial_img = np.sum(img, axis=2)
        im = axs[i, 0].imshow(spatial_img, cmap='viridis')
        axs[i, 0].set_title(f'{title} (Sum along energy axis)')
        axs[i, 0].set_xlabel('X axis')
        axs[i, 0].set_ylabel('Y axis')
        axs[i, 0].plot(pixel_x, pixel_y, 'r+', markersize=10)  # Mark the selected pixel
        plt.colorbar(im, ax=axs[i, 0])

    # Plot spectra for the selected pixel
    spectra = [actual_image[pixel_x, pixel_y, :],
               cropped_img[pixel_x, pixel_y, :],
               prediction_np[pixel_x, pixel_y, :]]

    for i, (spectrum, title) in enumerate(zip(spectra, titles)):
        energy_values = np.linspace(energy_range[0], energy_range[1], len(spectrum))
        axs[i, 1].plot(energy_values, spectrum)
        axs[i, 1].set_title(f'{title} Spectrum at pixel ({pixel_x}, {pixel_y})')
        axs[i, 1].set_xlabel('Energy (eV)')
        axs[i, 1].set_ylabel('Intensity')

    plt.tight_layout()
    
    return fig

def objective(trial, data, device):
    SIZE = 200
    energy_range = (900, 1100)
    
    # Define the hyperparameters to be tuned
    hyperparameters = {
        'sigma': trial.suggest_float('sigma', 0.1, 3.0),
        'epochs': trial.suggest_int('epochs', 50, 1000),
        'clip_value': trial.suggest_float('clip_value', 0.1, 5.0),
        'kl_weight': trial.suggest_float('kl_weight', 0.1, 2.0),
        'latent_dim': trial.suggest_int('latent_dim', 16, 512, step = 4),
        'xy_window': trial.suggest_int('xy_window', 1, 7, step=1),
    }
    
    model = CVAE3D(hyperparameters['latent_dim'], SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    final_loss, actual_image, train_image_viz = train_model(model, data, hyperparameters, device, optimizer, scheduler)
    
    # Visualize and save image
    fig = visualize_inference(model, actual_image, train_image_viz, energy_range, pixel_x=100, pixel_y=100)
    
    # Create filename based on trial number
    filename = f"trial_{trial.number}"
    fig.savefig(f"images/{filename}_images.jpg")
    
    del model
    torch.cuda.empty_cache()
    
    return final_loss

def run_optimization(rank, n_trials, data, result_queue):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data, device), n_trials=n_trials)
    
    result_queue.put((rank, study.best_trial))

def initialize_rankings_file(rankings_file):
    if not os.path.exists(rankings_file):
        with open(rankings_file, 'w') as f:
            json.dump([], f)
            
def update_rankings(trial, rankings_file, lock):
    with lock:
        if os.path.exists(rankings_file):
            with open(rankings_file, 'r') as f:
                rankings = json.load(f)
        else:
            rankings = []
        
        rankings.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params
        })
        
        rankings.sort(key=lambda x: x['value'])
        
        with open(rankings_file, 'w') as f:
            json.dump(rankings, f, indent=2)

def hyperparameter_tuning(data, n_trials=100):
    n_gpus = torch.cuda.device_count()
    trials_per_gpu = n_trials // n_gpus
    
    result_queue = mp.Queue()
    lock = Lock()
    rankings_file = "hyperparameter_rankings.json"

    processes = [
        mp.Process(target=run_optimization, args=(i, trials_per_gpu, data, result_queue))
        for i in range(n_gpus)
    ]

    for p in processes:
        p.start()

    # Main thread handles updating the rankings file
    completed_trials = 0
    while completed_trials < n_trials:
        rank, best_trial = result_queue.get()
        update_rankings(best_trial, rankings_file, lock)
        completed_trials += trials_per_gpu
        print(f"GPU {rank} completed. Total progress: {completed_trials}/{n_trials} trials")

    for p in processes:
        p.join()

    # Final analysis
    with open(rankings_file, 'r') as f:
        final_rankings = json.load(f)

    # Print top 10 results
    print("\nTop 10 Hyperparameter Combinations:")
    for i, result in enumerate(final_rankings[:10], 1):
        print(f"{i}. Loss: {result['value']:.4f}")
        for key, value in result['params'].items():
            print(f"   {key}: {value}")
        print()

    best_hyperparameters = final_rankings[0]['params']
    best_loss = final_rankings[0]['value']
    print(f"\nBest hyperparameters: {best_hyperparameters}")
    print(f"Best loss: {best_loss}")

    return best_hyperparameters

def train_model(model, data, hyperparameters, device, optimizer, scheduler):
    SIZE = 200
    energy_range = (900, 1100)
    batch_size = 32  # You can adjust this based on your GPU memory

    # Preprocess data
    train_image_viz, processed_data = preprocess_3d_images(
        data, 
        size=SIZE, 
        sigma=hyperparameters['sigma'], 
        energy_range=energy_range, 
        xy_window=hyperparameters['xy_window']
    )
    
    train_dataset = TensorDataset(torch.tensor(processed_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    num_examples_to_generate = 1
    test_sample = next(iter(train_loader))[0][:num_examples_to_generate].to(device)

    model.train()
    
    for epoch in range(hyperparameters['epochs']):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            mean, logvar = model.encode(x)
            z = model.reparameterize(mean, logvar)
            x_recon = model.decode(z)
            
            # Compute loss
            recon_loss = nn.MSELoss(reduction='sum')(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + hyperparameters['kl_weight'] * kl_loss
            
            # Backward pass and optimize
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['clip_value'])
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        # Average losses
        avg_loss = epoch_loss / len(train_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
        avg_kl_loss = epoch_kl_loss / len(train_loader.dataset)
        
        # Step the scheduler
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}], "
                  f"Loss: {avg_loss:.4f}, "
                  f"Recon Loss: {avg_recon_loss:.4f}, "
                  f"KL Loss: {avg_kl_loss:.4f}")
    
    return avg_loss, data, train_image_viz
    
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

# %%
def main():
    mp.set_start_method('spawn', force=True)
    print("Starting hyperparameter tuning process")
    dm4_file = '/home/ssulta24/VCAE/data/images_3D/BFO_a-0090 (dark ref corrected).dm3'
    data = load_dm4_data(dm4_file)
    best_hyperparameters = hyperparameter_tuning(data, n_trials=600)
    print("Hyperparameter tuning completed.")
    print(f"Best hyperparameters: {best_hyperparameters}")

if __name__ == "__main__":
    main()
