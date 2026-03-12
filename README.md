# Heterogeneous MOE for Diffusion models
# to do list:
    1. add zeta scheduler, balance between exploration and expolitation #DONE
    2. add CLIP text embeddings for text_guided diffusion #DONE
    3. data augmentation pipeline
        3.1. search for the most effective pipeline first

    4. Optional: add Post_hoc EMA power function
    5. add Stablility VAE encoder for latent diffusion. #DONE
    6. plots for: # DONE
        6.1. every loss VS training time 
        6.2. expert specialization graph
        6.3. FID score VS Number of function ->(denoiser) evaluations.
        6.4. max and mean weights for training stability (after training)
    
    7. Router's mask pattern generator for expert specialization #DONE
    8. EDM sampler for generating images (2nd order solver) #DONE
    9. implement lr_scheduler in utils.py #DONE