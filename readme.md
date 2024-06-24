# Archetypal Analysis for Identification of New Substages in Sleep

## Motivation
Preclinical mice models are often used to study sleep disorders, since it is possible to mimic specific sleep phenotypes in mice by gene manipulation. This makes it possible to study sleep processes in a very controlled setting. In sleep studies, electroencephalogram (EEG), electromyogram (EMG), and video are recorded over 24 hours to measure brain activity, muscle activity and behavior. Based on these recordings, a trained sleep expert manually divides the sleep into Wake, NREM, and REM in windows of 4-seconds.

Sleep in mice is very heterogeneous, thus it has been hypothesized whether the current three sleep stages covers all the variability of sleep. Recent studies ([1] and [2]) have looked into whether substages of sleep can be identified with different unsupervised clustering approaches. Identification of such substages will not only lead to more accurate sleep study evaluation but may also enable identification of disease specific stages which could improve diagnosis tremendously. Thus, it still remains an open question whether sleep in mice can be divided into more substages.

## Data
Our dataset comprises EEG and EMG recordings from approximately 150 mice, including both healthy and narcoleptic mice. The data is provided as both raw signals and preprocessed features. The features include various EEG power bands—Delta (0.25–5 Hz), Theta (5–9 Hz), Alpha (9–12 Hz), Beta (12–20 Hz), and Gamma (20–50 Hz)—extracted from the frequency domain in 4-second windows. In addition to the typical sleep stages, the dataset for narcoleptic mice includes recordings of additional sleep stages such as "Atypical Cataplexy" and "Cataplexy," enabling a more nuanced analysis of sleep patterns affected by narcolepsy.

## Setup Instructions

### Prerequisites
Ensure Python and all required packages are installed by running:

```bash
pip install -r requirements.txt
```

### Configuration
Before executing main.py, configure your config.json with the correct data path. Start by creating a config.json file based on the config.sample.json templat
```json
{
    "data_path": "path/to/your/data.csv",
    "narcolepsy_path": "path/to/your/narcolepsy.csv"
}
```
Replace "path/to/your/data.csv" with the actual path to your dataset's CSV file.

To run the main.py script from the src directory, navigate to the home directory, sleepy_mice, and execute:
```bash
python src/main.py
```

## Acknowledgments
1. Vasiliki-Maria Katsageorgiou, Diego Sona, Matteo Zanotto, Glenda Lassi, Celina Garcia-Garcia, Valter Tucci, and Vittorio Murino, "A novel unsupervised analysis of electrophysiological signals reveals new sleep substages in mice," PLOS Biology, vol. 16, no. 5, pp. 1–23, May 2018.
2. Farid Yaghouby and Sridhar Sunderam, "Segway: A simple framework for unsupervised sleep segmentation in experimental EEG recordings.," MethodsX, 2016.
3. Adele Cutler and Leo Breiman, “Archetypal analysis,” Technometrics, vol. 36, no. 4, pp. 338–347, 1994.
4. Morten Mørup and Lars Kai Hansen, “Archetypal analysis for machine learning and data mining,” Neurocomputing, vol. 80, pp. 54–63, 2012.


