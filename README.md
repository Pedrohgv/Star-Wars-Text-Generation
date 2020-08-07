# Star-Wars-Text-Generation
This Project aims to train a RNN using texts from Wookiepedia.com and generate Star Wars related text.
All required python libraries are listed on the "requirements.txt" file, and can be installed with the command 
"pip install -r requirements.txt" from the terminal. The project consists of a Recurrent Neural Network, with LSTM blocks and using a Sequence-to-sequence and character level
approach, built to generate pieces of text based on a given title. All the data, consisting of a title and the first
sentence of the introduction of an article, was obtained from the
[Wookiepdia website](https://starwars.fandom.com/wiki/Main_Page) using web scrapping tools.

An API was also developed, though it cannot be used using the files from this repository since the final model exceeds the maximum allowed size for individual files on GitHub.

Due to lack of more data and hardware limitations, the model couldn't have the complexity needed for more accurate results; however, the final results were still interesting. This is the output of the model when the input "obi-wan-kenobi" is given:

<img src="images/API.png" alt="drawing" width="900" height="444"/>
