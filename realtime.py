"""GPU streaming adapter: rolling hypotheses, bounded buffers, durable transcript."""
import os,sys,time,threading,queue,json,math,collections,logging
from pathlib import Path
ROOT=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).parent
for variable,subdir in [('CUDA_CACHE_PATH','cuda-cache'),('HF_HOME','hf-cache'),('XDG_CACHE_HOME','cache'),('TEMP','temp'),('TMP','temp')]:
 path=ROOT/subdir;path.mkdir(exist_ok=True);os.environ[variable]=str(path)
_dll_handles=[]
paths=[ROOT/'cuda',Path(getattr(sys,'_MEIPASS',ROOT))/'cuda']
if not getattr(sys,'frozen',False):paths+=list((ROOT.parent/'WhisperLive-0.10.0/.venv/Lib/site-packages/nvidia').glob('*/bin'))
for path in paths:
 if path.exists():
  os.environ['PATH']=str(path)+os.pathsep+os.environ.get('PATH','');_dll_handles.append(os.add_dll_directory(str(path)))
import numpy as np
from scipy.signal import resample_poly
import pyaudiowpatch as pa
from faster_whisper import WhisperModel
import desktop
from tkinter import ttk,messagebox,filedialog

class Streaming:
 def __init__(self,engine,events,stop,language,terms='',session=None):
  self.engine=engine;self.events=events;self.stop=stop;self.language=language;self.terms=terms
  self.lock=threading.Lock();self.pending={};self.finals=collections.deque();self.capture_done=False;self.draft_text={};self.stats=collections.deque(maxlen=2000);self.context={};self.session=session
 def submit(self,key,name,sample,rate,final):
  item=(key,name,sample,rate,final,time.monotonic())
  with self.lock:
   if final:
    self.pending.pop(key,None)
    if len(self.finals)>=32:
     if self.session:np.save(self.session.parent/('unprocessed-'+str(time.time_ns())+'.npy'),sample)
     self.events.put(('status','识别严重积压，已停止采集并保存未处理音频'));self.stop.set()
    self.finals.append(item)
   else:self.pending[key]=item
 def capture(self,selected):
  streams=[];states=[]
  def callback_for(d):
   rate=int(d['defaultSampleRate']);channels=d['maxInputChannels'];state={'chunks':[],'n':0,'quiet':0,'last':0,'id':0,'voice':False};states.append((d,state))
   def callback(data,n,info,status):
    try:
     if self.stop.is_set():return(None,pa.paComplete)
     if status:self.events.put(('status','音频驱动报告溢出，请检查系统负载'))
     x=np.frombuffer(data,dtype=np.float32).reshape(-1,channels).mean(axis=1).copy();loud=float(np.sqrt(np.mean(x*x)))>.001
     if not state['voice'] and not loud:
      state['chunks'].append(x);state['n']+=n
      while len(state['chunks'])>1 and state['n']-len(state['chunks'][0])>=rate*.2:state['n']-=len(state['chunks'].pop(0))
      return(None,pa.paContinue)
     state['voice']=True;state['chunks'].append(x);state['n']+=n;state['quiet']=0 if loud else state['quiet']+n
     final=state['quiet']>=rate*.65 or state['n']>=rate*18
     if final or state['n']-state['last']>=rate*.8:
      self.submit((d['index'],state['id']),d['name'],np.concatenate(state['chunks']),rate,final);state['last']=state['n']
     if final:state.update(chunks=[],n=0,quiet=0,last=0,id=state['id']+1,voice=False)
     return(None,pa.paContinue)
    except Exception as e:self.events.put(('status','采集错误：'+str(e)));self.stop.set();return(None,pa.paAbort)
   return callback
  try:
   with pa.PyAudio() as p:
    try:
     for d in selected:
      try:streams.append(p.open(format=pa.paFloat32,channels=d['maxInputChannels'],rate=int(d['defaultSampleRate']),input=True,input_device_index=d['index'],frames_per_buffer=1024,stream_callback=callback_for(d)))
      except Exception as e:self.events.put(('status','无法采集 '+d['name']+': '+str(e)))
     if not streams:raise RuntimeError('没有成功打开的音频输出设备')
     self.events.put(('status','GPU 实时转写中 · 灰色文字会持续校正'))
     while not self.stop.wait(.1):
      if not any(s.is_active() for s in streams):raise RuntimeError('音频设备已断开，请重新选择设备')
    finally:
     for s in streams:
      try:s.stop_stream()
      finally:s.close()
     for d,state in states:
      if state['voice'] and state['chunks']:self.submit((d['index'],state['id']),d['name'],np.concatenate(state['chunks']),int(d['defaultSampleRate']),True)
  except Exception as e:self.events.put(('status',str(e)));self.stop.set()
  finally:self.capture_done=True
 def decode(self):
  while True:
   with self.lock:
    item=self.finals.popleft() if self.finals else self.pending.pop(next(iter(self.pending))) if self.pending else None
   if item is None:
    if self.capture_done:break
    time.sleep(.02);continue
   key,name,sample,rate,final,stamp=item;start=time.monotonic()
   factor=math.gcd(rate,16000);audio=resample_poly(sample,16000//factor,rate//factor).astype(np.float32)
   prompt=(self.terms[:300]+' '+self.context.get(name,'')[-180:]).strip()
   segs,_=self.engine.transcribe(audio,language=self.language,task='transcribe',beam_size=5 if final else 3,temperature=0,condition_on_previous_text=False,initial_prompt=prompt or None,hotwords=self.terms[:300] or None,vad_filter=final,word_timestamps=False)
   text=''.join(s.text for s in segs).strip();elapsed=time.monotonic()-start;age=time.monotonic()-stamp
   self.stats.append({'seconds':len(audio)/16000,'decode':elapsed,'age':age,'final':final})
   if final:
    self.context[name]=text[-180:]
    if text and self.session:
     with self.session.open('a',encoding='utf-8') as f:f.write(text+'\n');f.flush()
   self.draft_text.pop(str(key),None) if final else self.draft_text.update({str(key):text})
   if self.session:
    draft=self.session.with_suffix('.draft.json');tmp=draft.with_suffix('.tmp');tmp.write_text(json.dumps(self.draft_text,ensure_ascii=False),encoding='utf-8');tmp.replace(draft)
   self.events.put(('hypothesis',(key,text,final)))
   self.events.put(('latency',f'GPU · 本次计算 {elapsed:.2f}s · 处理延迟 {age:.2f}s'))
   if age>3:self.events.put(('status','当前处理延迟超过 3 秒，请只选择会议使用的输出设备'))

class MeetingApp(desktop.App):
 def __init__(self,root):
  tk=desktop.tk;self.root=root;self.events=queue.Queue();self.stop_event=threading.Event();self.running=False;self.engine=None
  self.terms_value='';self.session=None;self.drafts={};self.committed=collections.deque(maxlen=400)
  self.settings=ROOT/'preferences.json'
  try:prefs=json.loads(self.settings.read_text(encoding='utf-8'));self.terms_value=prefs.get('terms','')
  except Exception:prefs={}
  root.title('WhisperMeeting · 实时会议工作台');root.geometry('1180x800');root.minsize(960,760);root.configure(bg='#0b1220')
  style=ttk.Style(root);style.theme_use('clam')
  style.configure('TCombobox',fieldbackground='#1c2b40',background='#24364e',foreground='#e8eef8',arrowcolor='#75dfc7',padding=8)
  style.map('TCombobox',fieldbackground=[('readonly','#1c2b40')],foreground=[('readonly','#e8eef8')])
  def label(parent,text,size=11,color='#92a5bf',**kw):return tk.Label(parent,text=text,font=('Microsoft YaHei UI',size),bg=parent.cget('bg'),fg=color,**kw)
  def button(parent,text,cmd,primary=False):return tk.Button(parent,text=text,command=cmd,font=('Microsoft YaHei UI',11,'bold'),bg='#63dec0' if primary else '#23334a',fg='#08241e' if primary else '#dde8f7',activebackground='#8aead3',activeforeground='#08241e',relief='flat',bd=0,padx=16,pady=8,cursor='hand2',disabledforeground='#62718a')
  header=tk.Frame(root,bg='#0b1220',padx=28,pady=18);header.pack(fill='x')
  label(header,'WHISPER / MEETING',10,'#63dec0').pack(anchor='w');label(header,'专注对话，让文字跟上。',24,'#f0f5ff').pack(anchor='w',pady=(6,0))
  content=tk.Frame(root,bg='#0b1220');content.pack(fill='both',expand=True,padx=24,pady=(0,16))
  side=tk.Frame(content,bg='#121e30',width=280,padx=20,pady=22);side.pack(side='left',fill='y',padx=(0,18));side.pack_propagate(False)
  label(side,'会议设置',15,'#edf4ff').pack(anchor='w',pady=(0,10))
  label(side,'音频来源',10).pack(anchor='w');self.choice=ttk.Combobox(side,state='readonly',width=24);self.choice.pack(fill='x',pady=(8,8))
  self.refresh=button(side,'刷新设备',self.reload);self.refresh.pack(fill='x',pady=(0,10))
  label(side,'会议语言',10).pack(anchor='w');self.lang=ttk.Combobox(side,state='readonly',values=['中文','English','自动检测']);self.lang.current(0);self.lang.pack(fill='x',pady=(8,10))
  self.terms_button=button(side,'人名 / 专业术语',self.set_terms);self.terms_button.pack(fill='x')
  label(side,'开始前设置，减少专有名词误识别。',9,wraplength=235,justify='left').pack(anchor='w',pady=(8,10))
  self.start=button(side,'●  开始转写',self.begin,True);self.start.pack(fill='x',pady=(0,10));self.end=button(side,'■  停止并保存',self.stop);self.end.configure(state='disabled');self.end.pack(fill='x')
  button(side,'导出原文',self.export).pack(fill='x',pady=(12,8));button(side,'打开会议记录',self.open_sessions).pack(fill='x')
  label(side,'本机处理 · D 盘自动保存',9,'#63dec0').pack(side='bottom',anchor='w')
  panel=tk.Frame(content,bg='#121e30',padx=24,pady=18);panel.pack(side='left',fill='both',expand=True)
  top=tk.Frame(panel,bg='#121e30');top.pack(fill='x');label(top,'实时字幕',16,'#edf4ff').pack(side='left');label(top,'灰色为待校正字幕',10).pack(side='right')
  self.status=tk.StringVar(value='就绪 · 选择音频设备，开始会议');tk.Label(panel,textvariable=self.status,font=('Microsoft YaHei UI',10),bg='#121e30',fg='#63dec0',anchor='w',wraplength=650,justify='left').pack(fill='x',pady=(10,16))
  area=tk.Frame(panel,bg='#121e30');area.pack(fill='both',expand=True)
  self.text=tk.Text(area,wrap='word',height=8,font=('Microsoft YaHei UI',17),bg='#121e30',fg='#edf3ff',insertbackground='#63dec0',selectbackground='#275d65',relief='flat',bd=0,padx=4,pady=8,spacing1=4,spacing3=12)
  self.text.pack(side='left',fill='both',expand=True);scroll=ttk.Scrollbar(area,command=self.text.yview);scroll.pack(side='right',fill='y');self.text.configure(yscrollcommand=scroll.set)
  self.text.tag_configure('draft',foreground='#8596af');self.text.bind('<Key>',lambda e:None if (e.state&4 and e.keysym.lower() in ['c','a']) else 'break')
  foot=tk.Frame(panel,bg='#121e30');foot.pack(fill='x',pady=(12,0));label(foot,'字幕字号',9).pack(side='left');tk.Scale(foot,from_=13,to=28,orient='horizontal',showvalue=False,command=lambda v:self.text.configure(font=('Microsoft YaHei UI',int(v))),bg='#121e30',fg='#edf3ff',highlightthickness=0,troughcolor='#23334a',activebackground='#63dec0',length=140,variable=tk.IntVar(value=17)).pack(side='left',padx=8)
  self.live_metric=tk.StringVar(value='GPU 加速 · 模型将在开始时预热');tk.Label(root,textvariable=self.live_metric,bg='#0b1220',fg='#92a5bf',font=('Microsoft YaHei UI',10),anchor='w',padx=28,pady=12).pack(side='bottom',fill='x',before=content)
  self.reload();root.after(80,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)
 def open_sessions(self):
  path=ROOT/'sessions';path.mkdir(exist_ok=True);os.startfile(path)
 def set_terms(self):
  from tkinter.simpledialog import askstring
  value=askstring('会议术语','输入人名、公司名和专业术语，用逗号分隔：',initialvalue=self.terms_value)
  if value is not None:
   self.terms_value=value;self.settings.write_text(json.dumps({'terms':value},ensure_ascii=False),encoding='utf-8')
 def begin(self):
  if self.running:return
  self.terms_button.configure(state='disabled')
  self.session=ROOT/'sessions'/time.strftime('%Y%m%d-%H%M%S.txt');self.session.parent.mkdir(exist_ok=True);self.session.touch()
  self.drafts.clear();self.committed.clear();self.text.delete('1.0','end');super().begin()
 def run(self,selected,language):
  worker=None;streaming=None
  try:
   self.events.put(('status','加载 GPU 模型并预热，请稍候…'))
   if self.engine is None:
    self.engine=WhisperModel(str(ROOT/'model-turbo'),device='cuda',compute_type='int8_float16',local_files_only=True,num_workers=1)
    list(self.engine.transcribe(np.zeros(16000,dtype=np.float32),language='zh',beam_size=1)[0])
   if self.stop_event.is_set():return
   streaming=Streaming(self.engine,self.events,self.stop_event,language,self.terms_value,self.session)
   worker=threading.Thread(target=streaming.capture,args=(selected,),daemon=True);worker.start();streaming.decode()
  except Exception as e:
   logging.exception('Meeting failed');self.events.put(('status','识别失败：'+str(e)))
  finally:
   self.stop_event.set()
   if worker:worker.join()
   if streaming and self.session:self.session.with_suffix('.metrics.json').write_text(json.dumps(list(streaming.stats)),encoding='utf-8')
   self.events.put(('done','已停止 · 原文已自动保存'))
 def poll(self):
  changed=False
  for _ in range(200):
   try:kind,value=self.events.get_nowait()
   except queue.Empty:break
   if kind=='hypothesis':
    key,text,final=value
    if final:
     self.drafts.pop(key,None)
     if text:self.committed.append(text)
    else:self.drafts[key]=text
    changed=True
   elif kind=='latency':self.live_metric.set(value)
   else:self.status.set(value)
   if kind=='done':
    self.terms_button.configure(state='normal');self.running=False;self.start['state']='normal';self.end['state']='disabled';self.refresh['state']='normal';self.choice['state']='readonly';self.lang['state']='readonly'
  if changed:
   self.text.delete('1.0','end');self.text.insert('end','\n'.join(self.committed)+'\n');self.text.insert('end','\n'.join(self.drafts.values()),'draft');self.text.see('end')
  self.root.after(80,self.poll)
 def export(self):
  path=filedialog.asksaveasfilename(defaultextension='.txt',filetypes=[('文本','*.txt')])
  if path:
   try:
    saved=self.session.read_text(encoding='utf-8') if self.session else ''
    Path(path).write_text(saved+'\n'.join(self.drafts.values()),encoding='utf-8-sig')
   except Exception as e:messagebox.showerror('导出失败',str(e))

if __name__=='__main__':
 import faulthandler
 crash=open(ROOT/'meeting-crash.log','a',encoding='utf-8');faulthandler.enable(crash)
 logging.basicConfig(filename=ROOT/'meeting.log',level=logging.INFO)
 if '--verify' in sys.argv:
  from faster_whisper.audio import decode_audio
  engine=WhisperModel(str(ROOT/'model-turbo'),device='cuda',compute_type='int8_float16',local_files_only=True)
  audio=decode_audio(sys.argv[sys.argv.index('--verify')+1],sampling_rate=16000)
  start=time.monotonic();segs,_=engine.transcribe(audio,language='en',beam_size=5);text=' '.join(s.text for s in segs)
  (ROOT/'exe-verification.json').write_text(json.dumps({'text':text,'seconds':time.monotonic()-start,'cuda':True}),encoding='utf-8')
  if 'computer' not in text.lower():raise RuntimeError('Verification transcript mismatch')
 else:MeetingApp(desktop.tk.Tk()).root.mainloop()


