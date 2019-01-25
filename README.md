# YouTube Project

*Disclaimer: This is the actual repository of our project. 
Please disregard the other repository of our group!*


## Searching YouTube Videos Using Image Recognition

The idea of this project was to make YouTube searches easier and more 
efficient for its users by adding a tool to search videos based on a picture. 
This way, a user who wants to find a YouTube video containing an object that is 
in his or her environment can simply take a picture using his smartphone camera 
and immediately obtain the results without having to type any text. 

The evaluation document of our project can be found in the root of this repository
and is called `evaluation-group01.pdf`. Moreover, the extensive project report is called
`report-group01.pdf`.

To make predicitons for existing images, or images taken with your webcam, no
images or annotations need to be downloaded. Predictions will be made using a
pretrained convolutional neural network, which is saved under the Models folder
of this repository. Training and validation images and annotations have to be 
downloaded and saved to disk only if a new model needs to be trained (e.g. when 
adding support for more object classes or changing the current network architecture).
In this case, the directories where the datasets were saved need to be set up via 
the script `paths.py`, which can be found in the ImageRecognition folder and the
*predict* flag should be set to False in `main.py`. For more details, please read
the *Implementation* section in our extensive project report.



## Running the Tool

_You must have Python 3.6 installed (not lower or higher!)_

- Install all the dependencies via the command line
    - `pip install -r requirements.txt`

- Run the tool in the IDE of your choice (tested in Spyder and PyCharm)
    - `Demo.py`


Enjoy!