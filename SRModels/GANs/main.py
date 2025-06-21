from SRModels.GANs.LPGANs import LaplacianPyramidGAN

if __name__ == "__main__":
    # Initialize the DCGAN model
    lpgan = LaplacianPyramidGAN()
    
    lpgan.train()