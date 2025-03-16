# Algorithm-to-Automatically-Generate-Minimalist-Numeral-Grammars
Combined Numeral Reinforcement Learning with CFG-to-MG algorithm to create 263 Minimalist Numeral Grammars in different languages.
See https://github.com/ikmMaierBTUCS/Numeral-Reinforcement-Learning and https://github.com/ikmMaierBTUCS/CFG-to-MG-algorithm

For setup, make sure that you have installed the packages ```numpy, num2words, sympy, diophantine, alphabet_detector, pandas```.

For a for a basic guided test you may run ```python Chatbot.py```.

In 'Code.py' you find the function numberset2MG. When passing a language* to the function, the numeral words of the language are learned and finally presented as a Minimalist Grammar in Stabler(1997) style.

*Pass a language by writing one the strings listed in 'Languages'. For some of the languages you also need to download 'Numeral.csv' and link its path to FILE_PATH_OF_LANGUAGESANDNUMBERS_DATA_CSV in the code.
