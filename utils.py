import numpy as np 
import cv2

expr_dict = {
    0:'Affraid',
    1:'Angry',
    2:'Disgusted',
    3:'Happy',
    4:'Neutral',
    5:'Sad',
    6:'Surprised'}

age_dict = {
    0:'< 20 years',
    1:'20-25 years',
    2:'25-30 years',
    3:'30-35 years',
    4:'35-40 years',
    5:'40-45 years',
    6:'45-50 years',
    7:'50-60 years',
    8:'> 60 years'}

def draw_expressions(labels, frame, x,y):
    cv2.rectangle(frame, (x,y-15), (x+220,y+20*7), (0,0,0), thickness=cv2.FILLED, lineType=8, shift=0)
    expr = labels[0].numpy()
    txt = ''
    for i, val in enumerate(expr):
        txt = '{}: {:.1f}%'.format(expr_dict[i],val*100)
        cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))
        y+=20
    return x,y 

def draw_smiley(labels, frame, x,y):
    best_expression = expr_dict[int(labels.max(dim=1)[1])]
    print(best_expression)
    if best_expression != 'Neutral':
        emoji_path = 'smiley/{}.png'.format(best_expression)
        overlay = cv2.imread(emoji_path)
        overlay = cv2.resize(overlay, (100,100), -1)
        rows,cols,channels = overlay.shape
        print(frame[y:y+rows, x:x+cols, :].shape)
        print(rows,cols,channels)
        overlay=cv2.addWeighted(frame[y:y+rows, x:x+cols, :],0,overlay,1,0)
        frame[y:y+rows, x:x+cols ] = overlay
    return frame

def draw_gender(labels, frame, x,y):
    cv2.rectangle(frame, (x,y-15), (x+220,y+20), (0,0,0), thickness=cv2.FILLED, lineType=8, shift=0)
    expr = labels[0].numpy()
    if expr[0] > expr[1]  :
        txt = 'Gender: Male ({:.1f}%)'.format(np.exp(expr[0])*100)
    else:
        txt = 'Gender: Female ({:.1f}%)'.format(np.exp(expr[1])*100)
    cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))

def draw_age(labels, frame, x,y):
    cv2.rectangle(frame, (x,y-15), (x+220,y+20), (0,0,0), thickness=cv2.FILLED, lineType=8, shift=0)
    mx = labels.max(dim=1)
    txt = 'Age: {} ({:.1f}%)'.format(age_dict[int(mx[1])], 100*np.exp(float(mx[0])))
    cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))