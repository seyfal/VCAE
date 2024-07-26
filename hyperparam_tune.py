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

import os
import sys 
import numpy as np
from datetime import datetime
from itertools import product
from operator import itemgetter
import json
from threading import Lock
import traceback

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

def create_loss_plot():
    """Create an empty Plotly figure for the loss plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Training Loss'))
    fig.update_layout(title='Training Loss',
                      xaxis_title='Epoch',
                      yaxis_title='Loss',
                      height=400,
                      width=800)
    return fig

def visualize_inference(device, model, input_image, data, energy_range=(900, 1100), x = 100, y = 100):
    model.eval()
    with torch.no_grad():
        input_tensor = input_image.unsqueeze(0).to(device)
        mean, logvar = model.encode(input_tensor)
        z = model.reparameterize(mean, logvar)
        prediction = model.decode(z, apply_sigmoid=True)

    # Set the pixel manually
    selected_pixel = (y, x) #! SET THE PIXEL MANUALLY 

    # Create two separate figures
    fig_images = make_subplots(rows=1, cols=2, subplot_titles=('Input Image', 'Prediction'))
    fig_spectra = make_subplots(rows=1, cols=2, subplot_titles=('Input Spectral Graph', 'Predicted Spectral Graph'))

    # Input Image
    middle_slice_input = input_image[0, :, :, input_image.shape[2] // 2].cpu().numpy()
    fig_images.add_trace(go.Heatmap(z=middle_slice_input, colorscale='Viridis', showscale=False), row=1, col=1)
    fig_images.add_trace(go.Scatter(x=[selected_pixel[1]], y=[selected_pixel[0]], mode='markers', 
                                    marker=dict(color='red', size=10), showlegend=False), row=1, col=1)

    # Prediction
    middle_slice_prediction = prediction[0, 0, :, :, prediction.shape[2] // 2].cpu().numpy()
    fig_images.add_trace(go.Heatmap(z=middle_slice_prediction, colorscale='Viridis', showscale=False), row=1, col=2)
    fig_images.add_trace(go.Scatter(x=[selected_pixel[1]], y=[selected_pixel[0]], mode='markers', 
                                    marker=dict(color='red', size=10), showlegend=False), row=1, col=2)

    # Update layout to ensure images are not distorted
    fig_images.update_layout(
        height=600,
        width=1200,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        yaxis2=dict(scaleanchor="x2", scaleratio=1)
    )

    # Input Spectral Graph
    input_spectrum = data[selected_pixel[0], selected_pixel[1], :]
    x_energy = np.linspace(energy_range[0], energy_range[1], input_spectrum.shape[0])
    fig_spectra.add_trace(go.Scatter(x=x_energy, y=input_spectrum), row=1, col=1)

    # Predicted Spectral Graph
    predicted_spectrum = prediction[0, 0, selected_pixel[0], selected_pixel[1], :].cpu().numpy()
    fig_spectra.add_trace(go.Scatter(x=x_energy, y=predicted_spectrum), row=1, col=2)

    fig_spectra.update_layout(
        height=500,
        width=1200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig_images, fig_spectra

def run_hyperparameter_tuning(rank, hyperparameter_combinations, data, result_queue, rankings_file, lock):    
    SIZE = 200 
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Create images directory
    os.makedirs("images", exist_ok=True)
    
    results = []
    for i, hyperparameters in enumerate(hyperparameter_combinations):
        print(f"GPU {rank}: Training combination {i+1}/{len(hyperparameter_combinations)}")
        print(f"Hyperparameters: {hyperparameters}")
        
        model = CVAE3D(hyperparameters['latent_dim'], SIZE).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        final_loss, image, train_image_viz = train_model(model, data, hyperparameters, device, optimizer, scheduler)
            
        # Visualize and save image
        fig_images, fig_spectra = visualize_inference(device=device, model=model, input_image=image, data=train_image_viz)
                
        # Create filename based on hyperparameters
        filename = "_".join([f"{k}_{v}" for k, v in hyperparameters.items()])
        fig_images.write_image(f"images/{filename}_images.jpg")
        fig_spectra.write_image(f"images/{filename}_spectra.jpg")
        
        results.append((hyperparameters, final_loss))
        print(f"GPU {rank}: Final Loss: {final_loss}\n")
        
        # Update rankings file
        update_rankings((hyperparameters, final_loss), rankings_file, lock)
        
        del model
        torch.cuda.empty_cache()
    
    result_queue.put(results)
    
def initialize_rankings_file(rankings_file):
    if not os.path.exists(rankings_file):
        with open(rankings_file, 'w') as f:
            json.dump([], f)

def update_rankings(new_result, rankings_file, lock):
    with lock:
        try:
            # Read existing rankings
            try:
                with open(rankings_file, 'r') as f:
                    rankings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                rankings = []
            
            # Add new result
            rankings.append(new_result)
            
            # Sort rankings by loss (ascending order)
            rankings.sort(key=lambda x: x[1])
            
            # Write updated rankings
            with open(rankings_file, 'w') as f:
                json.dump(rankings, f, indent=2)
            
            print(f"Rankings updated successfully. Total entries: {len(rankings)}")
        
        except Exception as e:
            print(f"Error updating rankings: {str(e)}")
            print(traceback.format_exc())
            
def hyperparameter_tuning(data):    
    hyperparameter_space = {
        'sigma': [1, 2, 3],
        'epochs': [150, 300, 600],
        'clip_value': [0.1, 0.5, 1.0, 2.0],
        'kl_weight': [0.1, 0.5, 1.0, 1.5, 2.0],
        'latent_dim': [20, 32, 64, 128, 256],
        'xy_window': [3, 5, 7]
    }

    combinations = list(product(*hyperparameter_space.values()))
    combinations = [dict(zip(hyperparameter_space.keys(), combo)) for combo in combinations]
    
    # Split combinations between two GPUs
    split_point = len(combinations) // 2
    combinations_gpu0 = combinations[:split_point]
    combinations_gpu1 = combinations[split_point:]

    rankings_file = "hyperparameter_rankings.json"

    # Initialize rankings file
    if not os.path.exists(rankings_file):
        with open(rankings_file, 'w') as f:
            json.dump([], f)

    # Use a manager to create a shared lock
    with mp.Manager() as manager:
        lock = manager.Lock()
        result_queue = manager.Queue()

        processes = [
            mp.Process(target=run_hyperparameter_tuning, args=(0, combinations_gpu0, data, result_queue, rankings_file, lock)),
            mp.Process(target=run_hyperparameter_tuning, args=(1, combinations_gpu1, data, result_queue, rankings_file, lock))
        ]

        for p in processes:
            p.start()

        # Main thread handles updating the rankings file
        completed_count = 0
        total_combinations = len(combinations)
        while completed_count < total_combinations:
            result = result_queue.get()
            completed_count += len(result)
            print(f"Completed {completed_count}/{total_combinations} combinations")

        for p in processes:
            p.join()

    # Final analysis
    with open(rankings_file, 'r') as f:
        final_rankings = json.load(f)

    sorted_results = final_rankings

    # Print top 10 results
    print("\nTop 10 Hyperparameter Combinations:")
    for i, (params, loss) in enumerate(sorted_results[:10], 1):
        print(f"{i}. Loss: {loss:.4f}")
        for key, value in params.items():
            print(f"   {key}: {value}")
        print("")

    # Analyze impact of individual hyperparameters
    print("\nHyperparameter Impact Analysis:")
    for param in hyperparameter_space.keys():
        param_impact = {}
        for combo, loss in sorted_results:
            value = combo[param]
            if value not in param_impact:
                param_impact[value] = []
            param_impact[value].append(loss)
        
        print(f"\n{param}:")
        for value, losses in sorted(param_impact.items()):
            avg_loss = sum(losses) / len(losses)
            print(f"  Value: {value}, Average Loss: {avg_loss:.4f}")

    best_hyperparameters, best_loss = sorted_results[0]
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
    
    return avg_loss, test_sample[0], train_image_viz
    
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
    best_hyperparameters = hyperparameter_tuning(data)
    print("Hyperparameter tuning completed.")
    print(f"Best hyperparameters: {best_hyperparameters}")

if __name__ == "__main__":
    main()
