import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
import pandas as pd

frameworks = {
    "DiffEqFlux.jl":"https://docs.sciml.ai/DiffEqFlux/stable/",
    "TensorFlow Probability":"https://www.tensorflow.org/probability?hl=pt-br",
    "PyMC": "https://www.pymc.io/welcome.html",
    "Scipy": "https://scipy.org/",
    "PySPH": "https://pythonhosted.org/PySPH/overview.html",
    "GPy": "https://gpy.readthedocs.io/en/deploy/",
    "Diffeqpy":"https://pypi.org/project/diffeqpy/",
    "Physics-Informed Neural Operators":"https://arxiv.org/pdf/2111.03794",
    "Variational Autoencoders": "https://www.geeksforgeeks.org/variational-autoencoders/",
    "Generative Adversarial Networks":"https://arxiv.org/pdf/1611.01673",
    "Reinforcement Learninig (RL) para SciML":"https://arxiv.org/pdf/2503.18892",
    "Symbolic Regression": "https://github.com/MilesCranmer/PySR",
    "Data Assimilation": "https://arxiv.org/pdf/2406.00390",
    "Uncertainty Quantification (UQ)":"https://en.wikipedia.org/wiki/Uncertainty_quantification",
    "DeepXDE":"https://deepxde.readthedocs.io/en/latest/",
    "Surrogate Models":"https://smt.readthedocs.io/en/latest/index.html",
    "Neural Operators":"https://neuraloperator.github.io/dev/user_guide/neural_operators.html",
    "Neuromancer": "https://github.com/pnnl/neuromancer",
    "Simulai": "https://github.com/IBM/simulai/",
    "TFC": "https://github.com/leakec/tfc",
    "FEM/FVM/FDM":"https://www.machinedesign.com/additive-3d-printing/fea-and-simulation/article/21832072/whats-the-difference-between-fem-fdm-and-fvm",
    "Physics Informed Extreme Learning Machine": "https://arxiv.org/pdf/1907.03507",
    "Bayesian Physics Informed Neural Networks": "https://arxiv.org/pdf/2205.08304",
    "Quantum Physics Informed Neural Networks": "https://arxiv.org/pdf/2503.12244",
    "Graph PINN's": "https://arxiv.org/pdf/2211.05520",
    "DeepXDE": "https://github.com/lululxvi/deepxde",
    "NVidia Modulus": "https://github.com/NVIDIA/physicsnemo",
    "OpenFOAM": "https://doc.openfoam.com/2312/quickstart/",
    "PhyGeoNet": "https://github.com/Jianxun-Wang/phygeonet",
    "SciML Julia": "https://docs.sciml.ai/Overview/stable/",
    "Runge-Kutta PINNs": "https://runge-kutta-pinn-cmi-cmi-public-aaca66b00a12c20ef6731763c12508c.pages.desy.de/",
    "Google DeepMind GraphCast": "https://github.com/google-deepmind/graphcast",
    "Classical PINNs": "https://github.com/maziarraissi/PINNs",
    "XTFC": "https://arxiv.org/pdf/2005.10632",
    "ROM":"https://pymor.org/",
    "Molecular Dynamics":"https://en.wikipedia.org/wiki/Molecular_dynamics",
    }

data = {"reference":[],"content":[]}

for framework in frameworks:
    reference = frameworks[framework]
    if 'pdf' in reference:
        reference_text = ''
        request = requests.get(reference).content
        with open('temp.pdf','wb') as f:
            f.write(request)
        pdf = PyPDF2.PdfReader('temp.pdf')
        for page in pdf.pages:
            reference_text+=page.extract_text()
        os.remove('temp.pdf')
    else:
        request = requests.get(reference).text
        soup = BeautifulSoup(request, 'html.parser')
        reference_text = soup.text
    data["reference"].append(reference)
    data["content"].append(reference_text)

df = pd.DataFrame(data)
df.to_csv('references_data.csv')