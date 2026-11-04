import os,sys,time,threading,queue,math,json
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
import pyaudiowpatch as pa
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
ROOT=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).parent

def devices():
    with pa.PyAudio() as p:return list(p.get_loopback_device_info_generator())

def model():return WhisperModel(str(ROOT/'model-small'),device='cpu',compute_type='int8',local_files_only=True)

class App:
    def __init__(self,root):
        self.root=root;root.title('WhisperLive · 电脑音频转写');root.geometry('960x700')
        self.events=queue.Queue();self.stop_event=threading.Event();self.running=False;self.engine=None
        body=ttk.Frame(root,padding=24);body.pack(fill='both',expand=True)
        ttk.Label(body,text='电脑声音 → 实时文字',font=('Microsoft YaHei UI',24,'bold')).pack(anchor='w')
        ttk.Label(body,text='WASAPI 系统回环 · 本地识别 · 无需浏览器共享',padding=(0,10)).pack(anchor='w')
        row=ttk.Frame(body);row.pack(fill='x',pady=10)
        self.choice=ttk.Combobox(row,state='readonly',width=65);self.choice.pack(side='left')
        self.refresh=ttk.Button(row,text='刷新设备',command=self.reload);self.refresh.pack(side='left',padx=8)
        self.lang=ttk.Combobox(body,state='readonly',values=['中文','English','自动检测'],width=20);self.lang.current(0);self.lang.pack(anchor='w')
        bar=ttk.Frame(body);bar.pack(fill='x',pady=16)
        self.start=ttk.Button(bar,text='开始识别',command=self.begin);self.start.pack(side='left')
        self.end=ttk.Button(bar,text='停止',command=self.stop,state='disabled');self.end.pack(side='left',padx=8)
        ttk.Button(bar,text='导出 TXT',command=self.export).pack(side='left')
        self.status=tk.StringVar(value='就绪');ttk.Label(body,textvariable=self.status).pack(anchor='w')
        self.text=tk.Text(body,wrap='word',font=('Microsoft YaHei UI',12),padx=14,pady=14);self.text.pack(fill='both',expand=True,pady=12)
        ttk.Label(body,text='默认同时采集所有输出设备；新接入耳机后请停止并刷新设备。优先在停顿处分段，最长约 12 秒；Small 高精度模型。').pack(anchor='w')
        self.reload();root.after(100,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)
    def reload(self):
        try:self.devs=devices();self.choice['values']=['全部输出设备']+[d['name'] for d in self.devs];self.choice.current(0)
        except Exception as e:messagebox.showerror('音频设备',str(e))
    def begin(self):
        if self.running:return
        selected=self.devs if self.choice.current()==0 else [self.devs[self.choice.current()-1]]
        if not selected:messagebox.showerror('无法开始','没有可用的系统音频回环设备');return
        self.running=True;self.stop_event.clear();self.start['state']='disabled';self.end['state']='normal';self.refresh['state']='disabled';self.choice['state']='disabled';self.lang['state']='disabled'
        language={'中文':'zh','English':'en','自动检测':None}[self.lang.get()]
        threading.Thread(target=self.run,args=(selected,language),daemon=True).start()
    def capture(self,selected,audio):
        # PortAudio initialization, stream lifetime and termination have one owner.
        streams=[];buffers=[]
        def enqueue(d,chunks,rate):
            if not chunks:return
            sample=np.concatenate(chunks)
            if np.max(np.abs(sample))>0.0005:
                try:audio.put_nowait((d['name'],sample,rate))
                except queue.Full:self.events.put(('status','处理队列已满，部分音频已跳过'))
        def callback_for(d):
            state={'chunks':[],'count':0,'silence':0};buffers.append((d,state))
            channels=d['maxInputChannels'];rate=int(d['defaultSampleRate'])
            def callback(data,n,info,status):
                try:
                    if self.stop_event.is_set():return(None,pa.paComplete)
                    state['chunks'].append(np.frombuffer(data,dtype=np.float32).reshape(-1,channels).mean(axis=1).copy());state['count']+=n
                    state['silence'] = state['silence'] + n if np.sqrt(np.mean(state['chunks'][-1]**2)) < 0.003 else 0
                    if (state['count']>=rate*3 and state['silence']>=rate*.6) or state['count']>=rate*12:
                        enqueue(d,state['chunks'],rate);state['chunks']=[];state['count']=0;state['silence']=0
                    return(None,pa.paContinue)
                except Exception as e:
                    self.events.put(('status','音频回调错误：'+str(e)));return(None,pa.paAbort)
            return callback
        try:
            with pa.PyAudio() as p:
                try:
                    for d in selected:
                        if self.stop_event.is_set():break
                        try:
                            stream=p.open(format=pa.paFloat32,channels=d['maxInputChannels'],rate=int(d['defaultSampleRate']),input=True,input_device_index=d['index'],frames_per_buffer=1024,stream_callback=callback_for(d))
                            streams.append(stream);self.events.put(('status','正在采集：'+d['name']))
                        except Exception as e:self.events.put(('status','设备打开失败：'+d['name']+'：'+str(e)))
                    while streams and not self.stop_event.wait(.1):
                        if not any(stream.is_active() for stream in streams):break
                finally:
                    for stream in streams:
                        try:stream.stop_stream()
                        finally:stream.close()
                    for d,state in buffers:enqueue(d,state['chunks'],int(d['defaultSampleRate']))
        except Exception as e:self.events.put(('status','音频采集错误：'+str(e)))
    def run(self,selected,language):
        workers=[];context={}
        try:
            self.events.put(('status','正在加载本地 Small 模型…'))
            if self.engine is None:self.engine=model()
            audio=queue.Queue(maxsize=24)
            if self.stop_event.is_set():return
            workers=[threading.Thread(target=self.capture,args=(selected,audio),daemon=True)]
            for w in workers:w.start()
            while any(w.is_alive() for w in workers) or not audio.empty():
                try:name,sample,rate=audio.get(timeout=.2)
                except queue.Empty:continue
                divisor=math.gcd(rate,16000);sample=resample_poly(sample,16000//divisor,rate//divisor).astype(np.float32)
                segments,_=self.engine.transcribe(sample,language=language,vad_filter=True,beam_size=5,temperature=0.0,condition_on_previous_text=False,initial_prompt=context.get(name),vad_parameters={'min_silence_duration_ms':500,'speech_pad_ms':300})
                text=''.join(s.text for s in segments).strip()
                if text:
                    context[name]=text[-200:]
                    self.events.put(('text',text))
        except Exception as e:self.events.put(('status','识别错误：'+str(e)))
        finally:
            self.stop_event.set()
            for w in workers:w.join()
            self.events.put(('done','已停止'))
    def stop(self):self.stop_event.set();self.end['state']='disabled';self.status.set('正在停止并处理剩余音频…')
    def poll(self):
        while True:
            try:kind,value=self.events.get_nowait()
            except queue.Empty:break
            if kind=='text':self.text.insert('end',value+'\n');self.text.see('end')
            else:self.status.set(value)
            if kind=='done':
                self.running=False;self.start['state']='normal';self.end['state']='disabled';self.refresh['state']='normal';self.choice['state']='readonly';self.lang['state']='readonly'
        self.root.after(100,self.poll)
    def export(self):
        path=filedialog.asksaveasfilename(defaultextension='.txt',filetypes=[('文本','*.txt')])
        if path:
            try:Path(path).write_text(self.text.get('1.0','end'),encoding='utf-8-sig')
            except Exception as e:messagebox.showerror('导出失败',str(e))
    def close(self):
        if self.running:
            self.stop();self.root.after(200,self.close_when_done)
        else:self.root.destroy()
    def close_when_done(self):
        if self.running:self.root.after(200,self.close_when_done)
        else:self.root.destroy()

if __name__=='__main__':
    import logging,faulthandler
    log_file=open(ROOT/'desktop-crash.log','a',encoding='utf-8')
    faulthandler.enable(log_file)
    logging.basicConfig(filename=ROOT/'desktop.log',level=logging.INFO)
    if '--capture-test' in sys.argv:
        obj=App.__new__(App);obj.events=queue.Queue();obj.stop_event=threading.Event();obj.engine=None
        for cycle in range(3):
            obj.stop_event.clear()
            timer=threading.Timer(8,obj.stop_event.set);timer.start()
            obj.run(devices(),'en');timer.cancel()
        Path(ROOT/'capture-test.json').write_text(json.dumps(list(obj.events.queue),ensure_ascii=False),encoding='utf-8')
        sys.exit(0)

    if '--self-test' in sys.argv:
        m=model();segments,_=m.transcribe(np.zeros(16000,dtype=np.float32));list(segments)
        Path(ROOT/'self-test.json').write_text(json.dumps({'model':'ok','devices':devices()},ensure_ascii=False),encoding='utf-8')
    else:App(tk.Tk()).root.mainloop()
