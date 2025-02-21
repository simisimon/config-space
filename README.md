# config-space

We extract configuration options using CfgNet. 
Note that you need to install the `config-space` branch, since the extraction of config options differs compared to the main branch.
You can install CfgNet using pip: `pip install git+https://github.com/AI-4-SE/CfgNet.git@config-space`


# Scenarios

## 1. Evolution of the Configuration Space
- motivated by Xu et al. (Hey, You Have Given Me Too Many Knobs!)
- configuration problems impair the reliability of software projects
- one fundamental problem is the ever-increasing complextity of configuration the configuration space
- reflected by an increasing number of configuration options, configuration constraints, and dependencies
- each software projects encodes hundreds to thousands of configuration options, often in technology-specific configuration files with their own syntax and semantics, which come with their own constraints and introduce dependencies between options
- configuring software becomes therefore a complex and error-prone task, as it requires understanding the configuration space and the impact of each option on the software's behavior
- this situation is worsned by the fact that the configuration space evolves over time, as developers add, remove, or modify configuration options due to bug fixes, new features, security updates, or just software evolution

Oberservations
- the complexity of the configuration space of software projects evolves over time 
- increasing number of configuration options
- removing option at a much slower rate than adding new options


## 2. Change Frequency of Configuration Options
- motivated by Xu et al. (Hey, You Have Given Me Too Many Knobs!)
- as the number of configuration options constantly increases, the following questions arise:
  - how many options are changed and unchanged?
  - how often do developers change configuration options?
  - do developers change the same options over time?
  - what kind of options are changed the most?
  - what kind of options to not change at all once set?
- we provide quantitative insights about the whether so many options are actually needed and how developers interact with them
- we highlight whether technologies are possibly over-configured or under-configured, helping to understand the configuration space of specific technologies

## 3. Co-Evolutionary Changes


## 4. Dependency Violations


## 5. Developer-Artifact Networks
