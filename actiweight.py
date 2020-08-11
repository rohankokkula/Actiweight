import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
st.markdown("""
<style>
    body {color:black;background-color:white;font-family: montserrat;}
</style>
<body></body>""", unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown(
        f"""
<style>
    .sidebar-content {{padding: 1rem !important;background: white !important;color:black;font-family: montserrat;}}
    .btn-outline-secondary {{
    border-color: #000000;
    color: #000;
    border-radius: 2rem;
    border-width: medium;}}
</style>
""",
unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;margin-top:-110px;font-family: montserrat;font-size:50px;'>ACTIWEIGHT</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: center;' >Select Function</h1>", unsafe_allow_html=True)
selected_act = st.sidebar.selectbox('', ("What are Activation Functions?","Sigmoid / Logistic Function","Hyperbolic Tangent Function","Rectified Linear Unit","Leaky Rectified Linear Unit","Parametric Rectified Linear Unit","Exponential Rectified Linear Unit","Swish","SoftMax","Softplus","Maxout"))
if(selected_act=="What are Activation Functions?"):
    st.image("intro.gif")
    st.markdown(f"""<h1 style='text-align: center;'>What are Activation Functions?</h1><h2>An activation function is a very important feature of an artificial neural network , they basically decide whether the neuron should be activated or not.<br><br> These type of functions are attached to each neuron in the network, and determines whether it should be activated or not, based on whether each neuron’s input is relevant for the model's prediction.<br><br>Important use of any activation function is to introduce <b>non-linear</b> properties to our Network.<br><br>In simple term , <br>it calculates a "weighted sum(Wi)" of its input(xi), adds a bias and then decides whether it should be "fired" or not.</h2>""", unsafe_allow_html=True)
elif(selected_act=="Sigmoid / Logistic Function"):
    logistic = lambda h, beta: 1./(1 + np.exp(-beta * h))
    beta = st.sidebar.slider("beta", -1, 25,5)
    plt.figure(figsize=(7,3.5))
    hvals = np.linspace(-5, 5)
    plt.plot(hvals, logistic(hvals, beta),"r",markersize=3,linewidth=1)
    log = st.sidebar.slider("Input Value", -5, 5,1)
    v=logistic(log, beta)
    plt.plot(log, v,'go', markersize=5)
    st.sidebar.markdown(f"## Output: {round(v,6)}")
    plt.xticks(np.arange(-5, 6, step=1))
    plt.grid(True)
    st.pyplot()
elif(selected_act=="Hyperbolic Tangent Function"):
    plt.figure(figsize=(6,4))
    hyperbolic_tangent = lambda h: (np.exp(h) - np.exp(-h)) / (np.exp(h) + np.exp(-h))
    theta = st.sidebar.slider("theta value", -1, 25,5)
    hvals = np.linspace(-5, 5)
    plt.plot(hvals, hyperbolic_tangent(hvals*theta),"r",markersize=3,linewidth=1)
    tan = st.sidebar.slider("Input Value", -5, 5,1)
    v=hyperbolic_tangent(tan*theta)
    plt.plot(tan, v,'go', markersize=5)
    st.sidebar.markdown(f"## Output: {round(v,6)}")
    plt.xticks(np.arange(-5, 6, step=1))
    plt.grid(True)
    st.pyplot()
elif(selected_act=="Rectified Linear Unit"):
    st.markdown(f"<h1 style='text-align: center;'>Rectified Linear Unit Activation Function</h1><h1 style='text-align: center;'>output=max(0,input)</h1>", unsafe_allow_html=True)
    x = np.linspace(0, 10)
    plt.figure(figsize=(6,4))
    plt.plot(x, x + 0, linestyle='solid')
    plt.plot([-10,0], [0,0],'r')
    plt.plot([0,10], [0,0],color="black",linewidth=0.5)
    plt.plot([0,0], [-2,10],color="black",linewidth=0.5)
    y = st.sidebar.slider("Input Value", -10, 10,2)
    plt.xticks(np.arange(-10, 11, step=2))
    if(y<=0):
        plt.plot(0, 0,'ro', markersize=5)
        st.sidebar.markdown("## Output: 0")
    else:
        plt.plot(y, y,'go', markersize=5)
        st.sidebar.markdown(f"## Output: {y}")
    st.pyplot()
    st.markdown("""
        <h3 style=' color: black;background-color: rgba(100, 100, 100, 0.05);'>The rectified linear activation function is a piecewise linear function that 
        will output the input directly if is positive,
        otherwise, it will output zero. It has become the default activation function 
        for many types of neural networks because a model that uses it is easier 
        to train and often achieves better performance.<br>
        The function must also provide more sensitivity to the activation sum input and avoid easy saturation.<br><br><br>
        The Dying ReLU problem<br>when inputs approach zero, or are negative, the gradient of the function becomes zero, the network cannot perform backpropagation and cannot learn.</h3>""", unsafe_allow_html=True)
elif(selected_act=="Leaky Rectified Linear Unit"):
    st.markdown(f"<h1 style='text-align: center;'>Leaky Rectified Linear Unit Activation Function</h1><h1 style='text-align: center;'>output=max[0,(0.01*input)]</h1>", unsafe_allow_html=True)
    x = np.linspace(0, 10)
    plt.figure(figsize=(7,4))
    plt.plot(x, x + 0, linestyle='solid')
    plt.plot([-50,50], [0,0],color="black",linewidth=0.5)
    plt.plot([0,0], [-4,12],color="black",linewidth=0.5)
    y = st.sidebar.slider("Input Value", -100, 10,2)
    if(y<=0):
        new=0.01*y
        plt.plot([y,0], [new,0],'r')
        plt.plot(y,new,'ro', markersize=5)
        plt.text(-50,10,f"Output: {new}")
    else:
        plt.text(-50,10,f"Output: {y}")
        plt.plot(y, y,'go', markersize=5)
    st.pyplot()
elif(selected_act=="Parametric Rectified Linear Unit"):
    st.markdown(f"<h1 style='text-align: center;'>Parametric Rectified Linear Unit Activation Function</h1><h1 style='text-align: center;'>output=max[0,(&alpha;*input)]</h1>", unsafe_allow_html=True)
    x = np.linspace(0, 10)
    plt.figure(figsize=(7,4))
    plt.plot(x, x + 0, linestyle='solid')
    plt.plot([-50,50], [0,0],color="black",linewidth=0.5)
    plt.plot([0,0], [-2,12],color="black",linewidth=0.5)
    y = st.sidebar.slider("Input Value", -100, 10,2)
    if(y<=0):
        scale = st.sidebar.slider("Alpha Value", 0.01, 0.1,0.01)
        new=scale*y
        plt.plot([y,0], [new,0],'r')
        plt.plot(y,new,'ro', markersize=5)
        plt.text(-50,10,f"Output: {new}")
    else:
        plt.text(-50,10,f"Output: {y}")
        plt.plot(y, y,'go', markersize=5)
    st.pyplot()
elif(selected_act=="Exponential Rectified Linear Unit"):
    st.markdown(f"<h1 style='text-align: center;'>Exponential Rectified Linear Unit Activation Function</h1><h1 style='text-align: center;'>output=max[0,(&alpha;*(e<sup>input</sup>-1))]</h1>", unsafe_allow_html=True)
    plt.figure(figsize=(7,5))
    y = st.sidebar.slider("Input Value", -100, 10,5)
    if(y<=0):
        plt.plot([-50,10], [0,0],color="black",linewidth=0.5)
        plt.plot([0,0], [-3,1],color="black",linewidth=0.5)        
        alpha = st.sidebar.slider("Alpha Value", 1, 5,1)
        new=alpha*(np.exp(y)-1)
        plt.plot(np.linspace(alpha*(y), 0, -alpha*y), alpha*(np.exp(np.linspace(alpha*(y), 0, -alpha*y))-1),color="red",linewidth=1)
        plt.plot(y,new,'ro-', markersize=5)
        plt.yticks(np.arange(-3, 0, step=0.6)) 
        st.sidebar.markdown(f"## Output: {round(new,4)}")
    else:    
        plt.plot([y,0], [y + 0,0],'bo-',markersize=5,color="blue")
        plt.plot([-50,10], [0,0],color="black",linewidth=0.5)
        plt.plot([0,0], [-3,12],color="black",linewidth=0.5)
        plt.yticks(np.arange(0, 11, step=1))
        st.sidebar.markdown(f"## Output: {y}")
        plt.plot(y, y,'go', markersize=5)
    st.pyplot()
elif(selected_act=="Swish"):
    st.markdown(f"<h1 style='text-align: center;'>Swish Activation Function</h1><h1 style='text-align: center;'>output=max[0,(&alpha;*(e<sup>input</sup>-1))]</h1>", unsafe_allow_html=True)


    


    