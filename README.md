The files audio and transcribe are modifications of the original files audio/transcribe of whisper library , the modification was necessary to ensure that whisper works (it worked before only with local file path/numpy array and numpy array I could not properly arrange)
The modification was taken from official whisper library github, someone added that feature as a pull request.

In order for the code to work you need to replace audio and transcribe  with modified files  audio/transcribe  from github + connect google cloud sdk ( authorize and push the docker iamge ) 
Just decided to add this meaningless app for myself
