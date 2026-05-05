---
title: 'BciPy 2.0: Experiment Orchestration, Multimodal Support, and Simulations in Python.'
tags:

- Python
- Brain-Computer Interface
- BCI
- Electroencephalography
- EEG
- event-related potential
- ERP
- P300
- brain signal processing
- real-time systems
- experiment design
- assistive technology
- augmentative and alternative communication
- AAC
  
authors:

- name: Tab Memmott
  orcid: 0000-0001-6143-5057
  corresponding: true
  equal-contrib: true
  affiliation: "1, 2"
- name: Matthew Lawhead
  orcid: 0000-0003-0736-3587
  equal-contrib: true
  affiliation: 3
- name: Basak Celik
  orcid: 0000-0002-0912-5243
  equal-contrib: true
  affiliation: 4
- name: Niklas Smedemark-Margulies
  orcid: 0000-0002-4364-0273
  equal-contrib: true
  affiliation: 4
- name: Dylan Gaines
  orcid: 0000-0002-2747-7680
  equal-contrib: true
  affiliation: "5, 6"
- name: Daniel Klee
  orcid: 0000-0003-2992-7662
  equal-contrib: true
  affiliation: 1
- name: Carson Reader
  equal-contrib: true
  affiliation: 2
- name: Allen Zhang
  equal-contrib: true
  affiliation: 2
- name: Srikar Ananthoju
  equal-contrib: true
  affiliation: 4
- name: Keith Vertanen
  orcid: 0000-0002-7814-2450
  equal-contrib: true
  affiliation: 5

affiliations:

- name: Department of Neurology, Oregon Health & Science University, Portland, OR, USA
  index: 1
- name: Institute on Development and Disability, Oregon Health & Science University, Portland, OR, USA
  index: 2
- name: Oregon Clinical and Translational Research Institute, Oregon Health & Science University, Portland, OR, USA
  index: 3
- name: Department of Electrical and Computer Engineering, Northeastern University, Boston, MA, USA
  index: 4
- name: Department of Computer Science, Michigan Technological University, Houghton, MI, USA
  index: 5
- name: Department of Computer Science, Kennesaw State University, Marietta, GA, USA
  index: 6

date: 15 April 2026

bibliography: paper.bib

---

# Summary

Advances in Brain-Computer Interface (BCI) research require software that evolves alongside new scientific discoveries and experimental needs. BciPy 2.0 is a major update and expansion of the original BciPy 1.0, developed in response to recent progress in the field. This release addresses growing demands for multimodal integration, offline simulation, and reproducible experimental protocols, with a particular focus on communication BCIs (cBCIs). BciPy 2.0 offers robust support for multimodal data acquisition and fusion, advanced simulation capabilities, flexible task orchestration, and standardized data sharing—all within the Python ecosystem. The system prioritizes modularity and extensibility, incorporating features informed by current research trends. This manuscript provides a comprehensive overview of BciPy 2.0, including system architecture and practical usage examples.

# Statement of Need

Software is the foundation of BCI research, serving as the bridge between biosignals and real-time applications that enable computer-mediated control. Reliable and adaptable tools are crucial for improving system accuracy, reducing latency, and expanding functionality—especially for communication BCI (cBCI) applications. These improvements bring cBCIs closer to practical, real-world use, with significant implications for healthcare, accessibility, and human-computer interaction.

Current trends in BCI research include increased interest in multimodal signal acquisition and integration, as well as the use of advanced modeling techniques to improve classification and inference. The scientific community is also prioritizing data practices that follow the FAIR principles — Findable, Accessible, Interoperable, and Reusable [@Wilkinson:2016] — which BciPy 2.0 is built to support. Additionally, some dependencies and Python versions used in BciPy 1.0 are now deprecated or incompatible with modern tools, motivating the architectural changes in BciPy 2.0. Future releases will continue to enhance interoperability with popular scientific libraries and features, and provide expanded support for experiment management.

# State of the Field

A large portion of the noninvasive BCI field relies on custom-built software or established frameworks such as BCI2000 [@Schalk:2004] and OpenViBE [@Renard:2010], which provide broad support for applications like cursor control and virtual reality (VR) integration but are written in C++ and oriented toward general BCI paradigms rather than communication-specific workflows. More recent Python packages such as PyBCI [@Booth:2023], MetaBCI [@Mei:2024], and PyNoetic [@Singh:2025] offer EEG signal acquisition, artifact handling, and interface control, but lack integrated support for multimodal evidence fusion, language modeling, and offline simulation. BciPy occupies a distinct niche by focusing specifically on text input for communication BCIs, combining these capabilities within a single Python framework. Python's dominance in scientific computing and machine learning allows BciPy to leverage a broad ecosystem of libraries and tools, making it accessible to researchers who may not have extensive programming experience.

# Software Design

BciPy supports installation on the latest versions of macOS, Linux, and Windows, with step-by-step instructions provided in the documentation and reproducible builds verified through continuous integration with GitHub Actions. Each submodule includes its own `README.md`, runnable demos, and unit tests to help users get started. Users can interact with BciPy through the client interface, by importing the package in Python, or via the PyQt6-based GUI (`BCInterface.py`, see \autoref{fig:gui}). The choice of interface depends on the user's coding experience and the level of customization required for their experiments.

<!-- Add Figure1 from static/ -->
![**BciPy GUI.** The BciPy GUI can be used for editing or loading parameters, training a `SignalModel`, defining a new experiment (this provides another GUI), or running an experiment or `Task`.\label{fig:gui}](static/Figure1.png){width=85%}

Experiment parameters are defined in JSON format, with default templates available in `bcipy/parameters/`. These parameters can be edited directly in the JSON files or through the graphical parameter editor shown in \autoref{fig:paramedit}, allowing researchers to configure experiment conditions upfront and reduce input errors. Data collected with BciPy can be exported to multiple formats—including BDF, EDF, BrainVision, Brain Imaging Data Structure (BIDS), and MNE—for external analysis or sharing, with compression options to facilitate storage. These export capabilities support FAIR data principles and interoperability with common analysis tools.

BciPy leverages several scientific libraries to provide its core features, including PsychoPy, PyLSL, scikit-learn, transformers, NumPy, SciPy, Pandas, and MNE [@Peirce:2007; @Kothe:2025; @Wolf:2020; @VanDerWalt:2011; @Virtanen:2020; @McKinney:2011; @Gramfort:2014]. The full list of dependencies is maintained in the `pyproject.toml` file.


<!-- Add Figure2 from static/ -->
![**BciPy Parameter Editor.** The BciPy Parameter Editing GUI can be used for editing, saving, or searching a parameters file. This can help prevent input errors and facilitate defining parameters for experiment conditions upfront. If parameters are changed, a panel under Changed Parameters (shown above) will display with the parameter changed and what value it’s been updated to. This can help prevent accidental changes or debug issues with a set of parameters.\label{fig:paramedit}](static/Figure2.png)

# Research Impact Statement

The BciPy repository has made BCI research more accessible through a modular, extensible, real-time Python interface designed for practical use and reproducible experimentation. The software and accompanying documentation have been released publicly, allowing researchers to directly run the system and adapt the interface for their own BCI studies by adding new paradigms and processing methods. The repository includes example pipelines, standardized data handling utilities, and integration with common Python scientific libraries, which has helped lower the technical barrier for working with neural signals in real time.

Evidence of use is reflected in 76,000 estimated downloads (from PyPI) and 39 external forks that adapt the interface for related BCI experiments and prototyping workflows. The BciPy Python library has also been used internally by 18 developers and by collaborators to build and test closed-loop BCI applications, demonstrating that the interface is stable enough for real experimental setups rather than only proof-of-concept demonstrations. Since its initial public release [@Memmott:2021], BciPy has gained wide adoption in the BCI community, with 145 GitHub stars, approximately 30 citations across peer-reviewed and preprint venues. There are also more than 10 peer-reviewed publications that have used BciPy in control and clinical studies (a full list is maintained in the repository `README.md`). Early adoption of BciPy 2.0 is evident, with five published studies utilizing its redesigned architecture as of March 2026. The Python BCI ecosystem has also grown, with several complementary libraries released or updated [@Zhu:2024; @Booth:2023; @Singh:2025; @Mei:2024].

BciPy is positioned for near-term impact within the BCI community due to its emphasis on reproducibility, clear documentation, and compatibility with common hardware, software, and analysis tools. By providing a simple and extensible interface for neural data acquisition and control, our work helps accelerate rapid prototyping and experimental iteration in BCI research.

# BciPy 2.0 Overview

BciPy 2.0 is a major update to the original BciPy 1.0, with significant improvements in architecture, functionality, and usability. The following sections describe the key additions in detail: task and experiment orchestration, multimodal data acquisition and evidence fusion, and a new simulation module for offline evaluation of system parameters.

# Task & Experiment Support

BciPy manages the execution of experimental `Task`s using the `SessionOrchestrator` class (see \autoref{fig:orchestrator}), which ensures `Task`s are run in the correct order and that all data are properly persisted. Researchers can run individual `Task`s, such as `Calibration`, directly via the client, or define complete experimental protocols for reproducible studies.

Experiment protocols are specified in the `experiments.json` file as ordered sequences of `Task`s. The orchestrator reads this protocol, executes each `Task` sequentially, and logs all relevant information. Each `Task` writes its own logs to a dedicated subdirectory for easy tracking and analysis.

In addition to standard `Task`s, protocols can include `Action`s, which are lightweight subclasses of `Task`. `Action`s do not require display or data acquisition and are useful for simple steps such as prompting the researcher with a dialog or indicating experiment progress. For example, an `IntertaskAction` can be inserted to request input from the researcher before continuing to the next `Task` in the sequence.

<!-- Add Figure3 from static/ -->
![**Session Orchestration.** The `SessionOrchestrator` executes a sequence of `Task`s defined in an experiment protocol. Each `Task` is initialized with the current parameters and any data needed from previous `Task`s. The orchestrator manages the flow of data between `Task`s and ensures that each `Task` is executed in the correct order. The `SessionOrchestrator`, once initialized with parameters and optional metadata, is ready for `Task`s to be added using the `add_tasks()` or `add_task()` methods. These can be defined and loaded using the `experiment.json` and defined protocol or manually added to the `SessionOrchestrator`. The experiment can then be run using `execute()`. This method loops over `Task`s, providing all parameters and a log needed for operation. The `Task`s are then responsible for initializing any objects required for operation, such as the `DataAcquisitionClient`, `Display`, or `LanguageModel`. After each `Task` and the entire execution loop, the data persists on disk.\label{fig:orchestrator}](static/Figure3.png){width=50%}

# Multimodal Data Acquisition and Fusion

BciPy 2.0 introduces support for multimodal data acquisition and evidence fusion. The system can consider information from multiple devices when making typing decisions. The BciPy 1.0 data acquisition module supported TCP-based connections as well as connections through LabStreamingLayer (LSL) [@Kothe:2025]. After extensive testing, we leaned more heavily on LSL to support multimodal acquisition in BciPy 2.0. LSL is well-supported across the industry, with many devices providing compatible drivers. This decision allowed us to drastically simplify the acquisition module while increasing functionality.

The data acquisition module in BciPy has two primary responsibilities: passively recording streaming device data to disk for later processing, and querying data in real time for use in a typing task. BciPy 2.0 streamlines the query interface from BciPy 1.0, replacing the previous unconstrained session-wide data access with a more efficient, event-driven approach suited to multimodal workflows.

The approach for combining multimodal data sources is described briefly. Under the conditional independence assumption, a Bayesian probabilistic algorithm that fuses information from multiple sources of evidence is employed to achieve multimodal classification. The posterior probabilities for user intent ($\theta$) given biosignals data ($x_{1:s}$) are computed using Bayes' rule as follows:

$$
p(\theta \mid x_{1:s}) \propto p(\theta) \prod_{j=1}^{s} p(x_j \mid \theta) \tag{1},
$$

where $s$ denotes the number of sources ($s > 1$ for multimodal) and $p(\theta)$ is the class prior.

In addition to the EEG signal model developed in BciPy 1.0, BciPy 2.0 introduces a gaze model for classification of eye gaze trajectory data that can be acquired in parallel with the EEG data stream through an eye tracker. Both the EEG and gaze models inherit the same `SignalModel` structure and are compatible with the scikit-learn estimators API [@Pedregosa:2011].

The provided gaze model assumes positional and temporal dependence in gaze data. It is also assumed that the gaze trajectory ($x_g$) obeys a Gaussian Process distribution, that is:

$$
p(x_g \mid \theta) \sim \mathcal{GP}(\mu_{g,\theta}, \Sigma_g) \tag{2},
$$

where $\mu_{g,\theta}$ and $\Sigma_g$ are the multidimensional means and shared covariances corresponding to the class labels, respectively. Details of the multimodal fusion method for EEG and gaze data are further described in the repository.

BciPy 2.0 expands upon the language modeling capabilities of BciPy 1.0 and removes the language model (LM) module from the previous Docker image, opting instead for direct function calls in Python. The goal of the language model remains the same as the prior version—to accelerate the text input task by providing additional evidence to the system. The LM module takes the context, or the text that the user has written so far, and produces an initial likelihood distribution over the system's symbol set. This distribution can be used to present more likely characters to the user sooner, in a paradigm like RSVP Keyboard [@Orhan:2012], or simply to fuse with the evidence gathered from the user (e.g., EEG, gaze, etc.).

The underlying models that drive the predictions are modular, and custom LM classes can be created or modified to suit users' needs. BciPy 2.0 introduces inference by large language models (LLMs) via the TextSlinger API, which allows the use of causal transformer models from Hugging Face using the search algorithm from @Gaines:2025. Wrapper classes such as `CausalLanguageModelAdapter` handle the initialization of TextSlinger models for seamless integration with BciPy. In addition to the causal LLM, BciPy 2.0 supports TextSlinger's `NGramLanguageModel` class, which leverages the KenLM package [@Heafield:2011].

BciPy 2.0 also provides a `UniformLanguageModel` class, which returns an equal probability distribution among all characters in the symbol set. This is useful when researchers want to remove language model influence as a control condition or independent variable, and it serves as an example of the methods required to create custom `LanguageModel` subclasses.

# BciPy Simulator

A major advancement in BciPy 2.0 is the introduction of the simulator module. This module allows researchers to use previously recorded typing data to systematically evaluate how changes in parameters, signal models, and language models impact typing performance. The simulator supports tasks such as customizing experiment parameters, conducting large-scale comparisons of language model implementations, and testing multimodal evidence fusion strategies. Additionally, the simulator framework can be extended to train new EEG signal models.

As illustrated in \autoref{fig:simulator}, the simulator architecture consists of several key components:

- The `SimulatorTask` (e.g., `CopyPhrase`),
- A `TaskRunner` for managing multiple iterations,
- A `DataEngine` for loading and querying data samples,
- A `DataProcessor` for formatting data for classification, and
- A `Sampler` that selects samples from the `DataEngine` using user-defined strategies.

The simulator collects metrics for each run and summarizes across all runs to assess performance. The simulator provides both a graphical user interface for designing simulation parameters and input sources, as well as a command line interface for scripting and automation.

<!-- Add Figure4 from static/ -->
![**BciPy Simulator Architecture.** The BciPy Simulator consists of several components that work together to simulate a typing task using previously recorded data. The `SimulatorTask` defines the task to be performed, such as `CopyPhrase`. The `TaskRunner` manages the execution of multiple iterations of the task, collecting metrics for each run. The `DataEngine` loads and queries data samples from the provided dataset, while the `DataProcessor` prepares the data to match the input format required by the classification model. The `Sampler` draws samples from the `DataEngine` based on a user-selected sampling strategy, such as random sampling or sequential sampling. The `SignalModel` and `LanguageModel` are used to classify the sampled data and provide predictions, respectively. Finally, the collected metrics are summarized across all runs to evaluate performance.\label{fig:simulator}](static/Figure4.png)

# AI usage disclosure

The manuscript was written by the authors with the assistance of AI tools, including ChatGPT, Claude, Grammarly, and GitHub Copilot. The AI tools were used to help edit text and to assist with code review, documentation, testing, and formatting. The majority of BciPy core functionality was developed by the authors exclusively. The authors reviewed and edited all content generated by the AI tools to ensure accuracy and coherence.

# Acknowledgements

We’d like to thank those who helped throughout the refactor, including Aida Fakhry, Ian Jackson, Julia Gangemi, Tales Imbiriba, Emma Sombers, Georgios Stratis, David Smith, Shijia Liu, Barry Oken, Deniz Erdogmus, Betts Peters, and Melanie Fried-Oken. In addition, we thank Steven Bedrick for his architectural and general advice on this significant update. This work was supported by NIH R01DC009834 and NSF IIS-1750193. Authors report no conflicts of interest.

# References
<!-- Add references from paper.bib -->