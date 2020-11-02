#!/usr/bin/env python
# coding: utf-8

# In[17]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install pyaudio')


# ## webcam 화면 띄우기

# In[22]:


import cv2
 
cam = cv2.VideoCapture(0)
cam.set(3,1280) #CV_CAP_PROP_FRAME_WIDTH
cam.set(4,720) #CV_CAP_PROP_FRAME_HEIGHT
#cam.set(5,0) #CV_CAP_PROP_FPS
 
while True:
    ret_val, img = cam.read() # 캠 이미지 불러오기
    img = cv2.flip(img, 1) # 1은 좌우반전, 0은 상하반전
 
    cv2.imshow("Cam Viewer",img) # 불러온 이미지 출력하기
    if cv2.waitKey(1) == 27: # esc to quit
        break
        
cam.release()
cv2.destroyAllWindows() # 웹캠 화면 끄기


# ## 마이크 불러오기 및 저장하기
# ### 출처 : https://blog.naver.com/chandong83/221149828690

# In[18]:


import pyaudio
from six.moves import queue
import time

# 녹음용 값 
# 16khz
RATE = 16000
# 버퍼는 1600
CHUNK = int(RATE / 10)  # 100ms


# In[19]:


class MicrophoneStream(object):
    """마이크 입력 클래스"""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # 마이크 입력 버퍼 생성
        self._buff = queue.Queue()
        self.closed = True

    # 클래스 열면 발생함.
    def __enter__(self):
        # pyaudio 인터페이스 생성
        self._audio_interface = pyaudio.PyAudio()
        # 16비트, 모노로 마이크 열기
        # 여기서 _fill_buffer 함수가 바로 callback함수 인데
        # 실제 버퍼가 쌓이면 이곳이 호출된다.
        # 즉, _fill_buffer 마이크 입력을 _fill_buffer 콜백함수로 전달 받음
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )        
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        # 클래스 종료시 발생
        # pyaudio 종료
        self._audio_stream.stop_stream()
        self._audio_stream.close()

        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()
    
    # 마이크 버퍼가 쌓이면(CHUNK = 1600) 이 함수 호출 됨. 
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        # 마이크 입력 받으면 큐에 넣고 리턴
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    # 제너레이터 함수 
    def generator(self):
        #클래스 종료될 떄까지 무한 루프 돌림 
        while not self.closed:
            
            # 큐에 데이터를 기다림.
            # block 상태임.
            chunk = self._buff.get()

            # 데이터가 없다면 문제 있음
            if chunk is None:
                return

            # data에 마이크 입력 받기
            data = [chunk]

            # 추가로 받을 마이크 데이터가 있는지 체크 
            while True:
                try:
                    # 데이터가 더 있는지 체크
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    # 데이터 추가
                    data.append(chunk)
                except queue.Empty:
                    # 큐에 데이터가 더이상 없다면 break
                    break

            #마이크 데이터를 리턴해줌 
            yield b''.join(data)
# [END audio_stream]


# In[20]:


def main():
    # 마이크 열기 
    with MicrophoneStream(RATE, CHUNK) as stream:
        # 마이크 데이터 핸들을 가져옴 
        audio_generator = stream.generator()
        for i in range(1000):
            # 1000번만 마이크 데이터 가져오고 빠져나감.

            for x in audio_generator:
                # 마이크 음성 데이터
                print(x)            
            time.sleep(0.001)

if __name__ == '__main__':
    main()


# ## 또 다른 마이크 관련 코드(위에랑 무관하다!!?)
# ### 출처 : https://stackoverrun.com/ko/q/12821098

# In[23]:


"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).
"""

import pyaudio

CHUNK = 1024
WIDTH = 2
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   output=True,
                   frames_per_buffer=CHUNK)

print("* recording")

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
       data = stream.read(CHUNK)  #read audio stream
       stream.write(data, CHUNK)  #play back audio stream

print("* done")

stream.stop_stream()
stream.close()

p.terminate()


# ## Webcam & Audio 함께!
# ### 출처 : https://www.manongdao.com/q-850995.html

# In[25]:


get_ipython().system('pip install ffpyplayer')


# In[26]:


import cv2
import numpy as np
#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer
video_path="../L1/images/Godwin.mp4"
def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()
PlayVideo(video_path)


# ## Webcam & Audio 동기화
# ### 출처 : https://stackoverflow.com/questions/14140495/how-to-capture-a-video-and-audio-in-python-from-a-camera-or-webcam?rq=1

# In[29]:


import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os

class VideoRecorder():  

    # Video class based on openCV 
    def __init__(self):

        self.open = True
        self.device_index = 0
        self.fps = 6               # fps should be the minimum constant rate at which the camera can
        self.fourcc = "MJPG"       # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (640,480) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = "temp_video.avi"
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()


    # Video starts being recorded 
    def record(self):

#       counter = 1
        timer_start = time.time()
        timer_current = 0


        while(self.open==True):
            ret, video_frame = self.video_cap.read()
            if (ret==True):

                    self.video_out.write(video_frame)
#                   print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
                    self.frame_counts += 1
#                   counter += 1
#                   timer_current = time.time() - timer_start
                    time.sleep(0.16)
#                   gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
#                   cv2.imshow('video_frame', gray)
#                   cv2.waitKey(1)
            else:
                break

                # 0.16 delay -> 6 fps
                # 


    # Finishes the video recording therefore the thread too
    def stop(self):

        if self.open==True:

            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

        else: 
            pass


    # Launches the video recording function using a thread          
    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()





class AudioRecorder():


    # Audio class based on pyAudio and Wave
    def __init__(self):

        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):

        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break


    # Finishes the audio recording therefore the thread too    
    def stop(self):

        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()





def start_AVrecording(filename):

    global video_thread
    global audio_thread

    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()

    audio_thread.start()
    video_thread.start()

    return filename




def start_video_recording(filename):

    global video_thread

    video_thread = VideoRecorder()
    video_thread.start()

    return filename


def start_audio_recording(filename):

    global audio_thread

    audio_thread = AudioRecorder()
    audio_thread.start()

    return filename




def stop_AVrecording(filename):

    audio_thread.stop() 
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    video_thread.stop() 

    # Makes sure the threads have finished
    while threading.active_count() > 1:
        time.sleep(1)


#    Merging audio and video signal

    if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected

        print("Re-encoding")
        cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.avi"
        subprocess.call(cmd, shell=True)

        print("Muxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)

    else:

        print("Normal recording\nMuxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)

        print("..")




# Required and wanted processing of final files
def file_manager(filename):

    local_path = os.getcwd()

    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")

    if os.path.exists(str(local_path) + "/temp_video.avi"):
        os.remove(str(local_path) + "/temp_video.avi")

    if os.path.exists(str(local_path) + "/temp_video2.avi"):
        os.remove(str(local_path) + "/temp_video2.avi")

    if os.path.exists(str(local_path) + "/" + filename + ".avi"):
        os.remove(str(local_path) + "/" + filename + ".avi")


# In[ ]:




