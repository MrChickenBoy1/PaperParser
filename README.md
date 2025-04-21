# PaperParser

A tool that goes over previous past papers of major examinations. It is able to give past year questions and create new sample papers. 

__________________________________________________________________________________________________________

Version 0.0.0 - This is a super, duper, simple version of what I have in my mind. At this point, there's just one PDF and a basic LangChain implementation to extract the text of the PDF, and an Ollama local hosting to process the text. This was created just to get something working, but the end result will definitely be done using JSON files. My next plan is to create a JSON extractor program, but I may resort to manual labeling for now using ChatGPT. 

Version 0.0.1 - Progress: I added JSON parsing capabilities. Regrettably, I had to use Claude. The program works, but it's really slow and it doesn't seem to be traversing the whole document. In order to move forward, I'll have to spend some time researching on various methods. 


Version 0.1 - Pretty much got JSON parsing working, with LLM generating a filter. Had to put it under the rug because I wasn't able to keep up with the size. This is common with projects and I definitely fell for the trap of over-scoping. I learnt a lot, so I am glad.
