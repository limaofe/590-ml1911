import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
OPT_ALGO='BFGS'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; xcol=1; ycol=2;
# model_type="linear";   NFIT=2; xcol=1; ycol=2; 
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;

#READ FILE
with open(INPUT_FILE) as f:
    my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
    if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
    
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#EXTRACT AGE<18
if(model_type=="linear"):
    y=y[x[:]<18]; x=x[x[:]<18]; 

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
    if(model_type=="linear"):   return  p[0]*x+p[1]  
    if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#LOSS FUNCTION
def loss(p, xt, yt):
    global iterations,loss_train,loss_val,iteration

    #TRAINING LOSS
    yp=model(xt,p) #model predictions for given parameterization p
    training_loss=(np.mean((yp-yt)**2.0))  #MSE

    #VALIDATION LOSS
    yp=model(xv,p) #model predictions for given parameterization p
    validation_loss=(np.mean((yp-yv)**2.0))  #MSE

    #WRITE TO SCREEN
    #if(iteration==0):    print("iteration    training_loss    validation_loss") 
    #if(iteration%25==0): print(iteration,"    ",training_loss,"    ",validation_loss) 
    
    #RECORD FOR PLOTING
    loss_train.append(training_loss); loss_val.append(validation_loss)
    iterations.append(iteration); iteration+=1

    return training_loss


    
#INITIAL GUESS
po=np.random.uniform(0.1,1.,size=NFIT)

#OPTIMAZATION FUNCTION
def GD(objective, xt, yt, po, LR=0.001):
    dx=0.001                            #STEP SIZE FOR FINITE DIFFERENCE
    t=0                                  #INITIAL ITERATION COUNTER
    tmax=30000                            #MAX NUMBER OF ITERATION
    tol=10**-10                            #EXIT AFTER CHANGE IN F IS LESS THAN THIS 

    while(t<=tmax):
        t=t+1
        #print('iteration',t)
        #NUMERICALLY COMPUTE GRADIENT 
        df_dx=np.zeros(NFIT)
        for i in range(0,NFIT):
            dX=np.zeros(NFIT);
            dX[i]=dx; 
            xm1=po-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
            df_dx[i]=(objective(po, xt, yt)-objective(xm1, xt, yt))/dx
        #print(xi.shape,df_dx.shape)
        xip1=po-LR*df_dx #STEP 
    
        if(t%100==0):
            df=np.mean(np.absolute(objective(xip1, xt, yt)-objective(po, xt, yt)))
            print(t,"    ",po,"    ","    ",objective(po, xt, yt)) #,df) 
    
            if(df<tol):
                print("STOPPING CRITERION MET (STOPPING TRAINING)")
                break
    
        #UPDATE FOR NEXT ITERATION OF LOOP
        po=xip1
    return po

def GD_MOM(objective, xt, yt, po, LR=0.001, MOM=0.005):
    dx=0.001                            #STEP SIZE FOR FINITE DIFFERENCE
    t=0                                  #INITIAL ITERATION COUNTER
    tmax=30000                            #MAX NUMBER OF ITERATION
    tol=10**-10                            #EXIT AFTER CHANGE IN F IS LESS THAN THIS 

    while(t<=tmax):
        t=t+1
        #print('iteration',t)
        #NUMERICALLY COMPUTE GRADIENT 
        df_dx=np.zeros(NFIT)
        for i in range(0,NFIT):
            dX=np.zeros(NFIT);
            dX[i]=dx; 
            xm1=po-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
            df_dx[i]=(objective(po, xt, yt)-objective(xm1, xt, yt))/dx
        #print(xi.shape,df_dx.shape)
        xip1=po-LR*df_dx+dx*MOM #STEP 
    
        if(t%100==0):
            df=np.mean(np.absolute(objective(xip1, xt, yt)-objective(po, xt, yt)))
            print(t,"    ",po,"    ","    ",objective(po, xt, yt)) #,df) 
    
            if(df<tol):
                print("STOPPING CRITERION MET (STOPPING TRAINING)")
                break
    
        #UPDATE FOR NEXT ITERATION OF LOOP
        po=xip1
    return po

#OPTIMIZER FUNCTION
def optimizer(objective, po, algo='GD', LR=0.001, method='batch'):
    global xt, yt, xv, yv
    if algo=='GD':
        if method=='batch':
            res=GD(objective, xt, yt, po, LR=LR)
            return res
            
        if method=='minibatch':
            batch_size=0.5
            rand_indices = np.random.permutation(xt.shape[0])
            CUT1=int(batch_size*xt.shape[0]); 
            batch1_idx,  batch2_idx = rand_indices[:CUT1], rand_indices[CUT1:]
            x_b1=xt[batch1_idx]
            y_b1=yt[batch1_idx]
            po1=GD(objective, x_b1, y_b1, po, LR=LR)
            x_b2=xt[batch2_idx]  
            y_b2=yt[batch2_idx]
            res=GD(objective, x_b2, y_b2, po1, LR=LR)
            return res
        
        if method=='stochastic':
            for i in range(len(xt)):
                xi=xt[i]
                yi=yt[i]
                po=GD(objective, xi, yi, po, LR=LR)
            return po
    if algo=='GD+MOM':
        if method=='batch':
            res=GD_MOM(objective, xt, yt, po, LR=LR)
            return res
            
        if method=='minibatch':
            batch_size=0.5
            rand_indices = np.random.permutation(xt.shape[0])
            CUT1=int(batch_size*xt.shape[0]); 
            batch1_idx,  batch2_idx = rand_indices[:CUT1], rand_indices[CUT1:]
            x_b1=xt[batch1_idx]
            y_b1=yt[batch1_idx]
            po1=GD_MOM(objective, x_b1, y_b1, po, LR=LR)
            x_b2=xt[batch2_idx]  
            y_b2=yt[batch2_idx]
            res=GD_MOM(objective, x_b2, y_b2, po1, LR=LR)
            return res
        
        if method=='stochastic':
            for i in range(len(xt)):
                xi=xt[i]
                yi=yt[i]
                po=GD_MOM(objective, xi, yi, po, LR=LR)
            return po
            
            

#TRAIN MODEL USING SCIPY MINIMIZ 
# res = minimize(loss, po, method=OPT_ALGO, tol=1e-15);  popt=res.x



#SELECT ONE OF THE FOLLOWING CODE TO RUN by 'batch', 'minibatch' or 'stochastic'
#res = optimizer(loss, po, algo='GD', LR=0.001, method='batch')
#print("OPTIMAL PARAM:", res)

res = optimizer(loss, po, algo='GD', LR=0.001, method='minibatch')
print("OPTIMAL PARAM:", res)

#TRY DIFFERENT LEARNING RATE
#res = optimizer(loss, po, algo='GD', LR=0.003, method='minibatch')
#print("OPTIMAL PARAM:", res)

#res = optimizer(loss, po, algo='GD', LR=0.001, method='stochastic')
#print("OPTIMAL PARAM:", res)

#res = optimizer(loss, po, algo='GD+MOM', LR=0.001, method='batch')
#print("OPTIMAL PARAM:", res)

#res = optimizer(loss, po, algo='GD+MOM', LR=0.001, method='minibatch')
#print("OPTIMAL PARAM:", res)

#res = optimizer(loss, po, algo='GD+MOM', LR=0.001, method='stochastic')
#print("OPTIMAL PARAM:", res)


#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm, res))

#UN-NORMALIZE
def unnorm_x(x): 
    return XSTD*x+XMEAN  
def unnorm_y(y): 
    return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
    ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
    ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.legend()
    plt.show()

#PARITY PLOTS
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(model(xt,res), yt, 'o', label='Training set')
    ax.plot(model(xv,res), yv, 'o', label='Validation set')
    plt.xlabel('y predicted', fontsize=18)
    plt.ylabel('y data', fontsize=18)
    plt.legend()
    plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(iterations, loss_train, 'o', label='Training loss')
    ax.plot(iterations, loss_val, 'o', label='Validation loss')
    plt.xlabel('optimizer iterations', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.legend()
    plt.show()
    
# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND EVERYTHING THAT IT IS DOING