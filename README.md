# Predicting Optogenetic Cell Migration with Contrastive RL

The primary goal of this project is to create an unsupervised RL algorithm capable of predicting how cells containing OptoEGFR migrate when exposed to different light stimuli. By accurately modeling cell movement, we can enhance techniques in tissue engineering, such as guiding cells to specific configurations to accelerate wound healing or study tissue morphogenesis.

This project builds upon and collaborates with research conducted by **Suh et al.** in the **Toettcher Lab**, where OptoEGFR was introduced as a mechanism for large-scale optogenetic control of mammalian cells. The initial studies used a simplified boundary-flux model based on empirical data to output relative cell densities in response to illumination inputs. The original paper can be found here:

**Suh, K., Thornton, R., Farahani, P. E., Cohen, D., & Toettcher, J.** (2024). *Large-scale control over collective cell migration using light-controlled epidermal growth factor receptors*. bioRxiv. [https://doi.org/10.1101/2024.05.30.596676](https://doi.org/10.1101/2024.05.30.596676)

However, the boundary-flux model, while useful, overlooks important factors like:

- Preexisting cell densities
- Tissue stress and mechanics
- Inter-cell interactions
- Tissue dynamics like fluidization and density-driven jamming

These factors significantly impact cell migration and are essential for precise applications like wound healing.

Contrastive RL presents an opportunity to learn more nuanced representations of these dynamics, potentially outperforming the boundary-flux model by taking into consideration preexisting cell densities, cellular interactions, and tissue dynamics like fluidization and jamming.

## Requirements

- **Python 3.8** or higher
- **PyTorch 1.7** or higher
- Other dependencies as listed in `requirements.txt`

## Installation

1. **Create a virtual environment :**

   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```
