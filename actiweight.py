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
selected_act = st.sidebar.selectbox('', ("What are Activation Functions?","Sigmoid / Logistic Function","Hyperbolic Tangent Function","Rectified Linear Unit","Leaky Rectified Linear Unit","Parametric Rectified Linear Unit","Exponential Linear Unit","Swish Function","SoftMax Function","Softplus Function","Maxout Function"))
if(selected_act=="What are Activation Functions?"):
    st.image("intro.gif")
    st.markdown(f"""<h1 style='text-align: center;'>What are Activation Functions?</h1><h2>An activation function is a very important feature of an artificial neural network , they basically decide whether the neuron should be activated or not.<br><br> These type of functions are attached to each neuron in the network, and determines whether it should be activated or not, based on whether each neuron’s input is relevant for the model's prediction.<br><br>Important use of any activation function is to introduce <b>non-linear</b> properties to our Network.<br><br>In simple term , <br>it calculates a "weighted sum(Wi)" of its input(xi), adds a bias and then decides whether it should be "fired" or not.</h2>""", unsafe_allow_html=True)
elif(selected_act=="Sigmoid / Logistic Function"):
    st.sidebar.image('formulae/sigmoid.png',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/50a861269c68b1f1b973155fa40531d83c54c562',width=300)
    logistic = lambda h, beta: 1./(1 + np.exp(-beta * h))
    beta = st.sidebar.slider("Select value of β", -1, 25,1)
    plt.figure(figsize=(7,3.5))
    hvals = np.linspace(-5, 5)
    plt.plot(hvals, logistic(hvals, beta),"r",markersize=3,linewidth=1)
    log = st.sidebar.slider("Select value of x", -5, 5,1)
    v=logistic(log, beta)
    plt.plot(log, v,'go', markersize=5)
    st.sidebar.markdown(f"## Output: {round(v,6)}")
    plt.plot([-5,5], [0,0],color="black",linewidth=0.3)
    plt.plot([0,0], [1,-0.1],color="black",linewidth=0.3)
    plt.xticks(np.arange(-5, 6, step=1))
    plt.yticks(np.arange(0, 1, step=0.1))
    st.pyplot()
elif(selected_act=="Hyperbolic Tangent Function"):
    st.sidebar.image('formulae/tanh.png',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/b371c445bf1130914d25b1995d853ac0e27bc956',width=200)
    plt.figure(figsize=(7,3.5))
    hyperbolic_tangent = lambda h: (np.exp(h) - np.exp(-h)) / (np.exp(h) + np.exp(-h))
    theta = st.sidebar.slider("Select value of θ", -1, 25,1)
    hvals = np.linspace(-5, 5)
    plt.plot(hvals, hyperbolic_tangent(hvals*theta),"r",markersize=3,linewidth=1)
    tan = st.sidebar.slider("Select value of x", -5, 5,1)
    v=hyperbolic_tangent(tan*theta)
    plt.plot(tan, v,'go', markersize=5)
    st.sidebar.markdown(f"## Output: {round(v,6)}")
    plt.xticks(np.arange(-5, 6, step=1))
    plt.yticks(np.arange(-1, 1.2, step=0.5))
    plt.plot([-5,5], [0,0],color="black",linewidth=0.3)
    plt.plot([0,0], [-1,1],color="black",linewidth=0.3)
    st.pyplot()
elif(selected_act=="Rectified Linear Unit"):
    st.markdown(f"<h1 style='text-align: center;margin-top:-60px;'>Rectified Linear Unit Activation Function</h1>", unsafe_allow_html=True)
    st.sidebar.image('formulae/relu.png',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/e8723cef7eb5dedf4aa20e174ee281b76a6cbec4',width=300)
    x = np.linspace(0, 10)
    plt.figure(figsize=(6,4))
    plt.plot(x, x + 0, linestyle='solid')
    plt.plot([-10,0], [0,0],'r')
    plt.plot([0,10], [0,0],color="black",linewidth=0.5)
    plt.plot([0,0], [-2,10],color="black",linewidth=0.5)
    y = st.sidebar.slider("Select value of x", -10, 10,2)
    plt.xticks(np.arange(-10, 11, step=2))
    if(y<0):
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
    st.markdown(f"<h1 style='text-align: center;'>Leaky Rectified Linear Unit Activation Function</h1>", unsafe_allow_html=True)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/aaabce8985d074b5f4482f4efa327c7c61da3ca6',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/53ae28dca48ccda3b6ac4d5b0ecec23b70368784',width=300)
    x = np.linspace(0, 10)
    plt.figure(figsize=(7,4))
    plt.plot(x, x + 0, linestyle='solid')
    plt.plot([-100,10], [0,0],color="black",linewidth=0.5)
    plt.plot([0,0], [-1,10],color="black",linewidth=0.5)
    y = st.sidebar.slider("Select value of x", -100, 10,2)
    plt.xticks(np.arange(-100,11, step=10))
    plt.yticks(np.arange(-1,11, step=1))
    if(y<0):
        new=0.01*y
        plt.plot([y,0], [new,0],'r')
        plt.plot(y,new,'ro', markersize=5)
        st.sidebar.markdown(f"## Output: {new}")
    else:
        st.sidebar.markdown(f"## Output: {y}")
        plt.plot(y, y,'go', markersize=5)
    st.pyplot()
elif(selected_act=="Parametric Rectified Linear Unit"):
    st.markdown(f"<h1 style='text-align: center;'>Parametric Rectified Linear Unit Activation Function</h1>", unsafe_allow_html=True)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/387a2af979ccc6a29b62950e1efb7c3a86209ad7',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/51480bf026d4e3149f7c815fda04940663894791',width=300)
    plt.figure(figsize=(7,5))
    y = st.sidebar.slider("Select value of x", -100, 10,2)
    plt.plot([-60,30], [0,0],color="black",linewidth=0.5)
    plt.xticks(np.arange(-100,11, step=10))
    if(y<0):
        alpha = st.sidebar.slider("Select value of Alpha", 0.01, 1.0,0.01,step=0.1)
        new=alpha*y
        plt.plot([y,0], [new,0],'r')
        plt.plot([0,0], [-30,10],color="black",linewidth=0.5)
        plt.plot(y,new,'ro', markersize=5)
        st.sidebar.markdown(f"## Output: {new}")
        st.pyplot()
    else:
        plt.plot([0,0], [-30,10],color="black",linewidth=0.5)
        plt.plot([y,0], [y,0],'g')
        plt.plot(y,y,'go', markersize=5)
        st.sidebar.markdown(f"## Output: {y}")
        st.pyplot()
elif(selected_act=="Exponential Linear Unit"):
    st.markdown(f"<h1 style='text-align: center;'>Exponential Linear Unit Activation Function</h1>", unsafe_allow_html=True)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/5d740c6ed2015b0208a6945e53f10d89c11855b3',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/ad31fcdb29df53296c148a3af5c1b66bc7317a90',width=300)
    plt.figure(figsize=(7,5))
    y = st.sidebar.slider("Input Value", -100, 10,5)
    plt.plot([-60,30], [0,0],color="black",linewidth=0.5)
    if(y<0):
        alpha = st.sidebar.slider("Alpha Value", 1, 5,1)
        new=alpha*(np.exp(y)-1)
        plt.plot([0,0], [-5,10],color="black",linewidth=0.5)        
        x=np.linspace(y, 0, 100)
        z=alpha*(np.exp(x)-1)
        plt.plot(x, z,color="red",linewidth=1)
        plt.plot(y,new,'bo', markersize=5)
        st.sidebar.markdown(f"## Output: {round(new,4)}")
    else:    
        plt.plot([0,0], [-5,10],color="black",linewidth=0.5)
        plt.plot([y,0], [y,0],'g')
        plt.plot(y,y,'bo', markersize=5)
        st.sidebar.markdown(f"## Output: {y}")
    st.pyplot()
elif(selected_act=="Swish Function"):
    st.sidebar.image('formulae/swish.png',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/e1e28d510bf1a0d4f2ae34557bcd141dfa51063f',width=300)
    logistic = lambda h, beta: 1./(1 + np.exp(-beta * h))
    beta = st.sidebar.slider("Select value of β", -1, 25,1)
    plt.figure(figsize=(7,3.5))
    hvals = np.linspace(-5, 5)
    plt.plot(hvals, hvals*logistic(hvals, beta),"r",markersize=3,linewidth=1)
    log = st.sidebar.slider("Select value of x", -5.0, 5.0,0.5)
    v=log*logistic(log, beta)
    plt.plot(log, v,'go', markersize=5)
    st.sidebar.markdown(f"## Output: {round(v,6)}")
    plt.plot([-5,5], [0,0],color="black",linewidth=0.3)
    plt.plot([0,0], [5,-0.2],color="black",linewidth=0.3)
    plt.xticks(np.arange(-5, 6, step=1))
    plt.yticks(np.arange(-0.5, 5.5, step=0.5))
    st.pyplot()
elif(selected_act=="Softplus Function"):
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/f21f3d1e2c67c5c2d2085e384512bc737c8e14af',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/7f9e624f08baaf2b5886fc69c1162b8caf79f622',width=300)
    soft = lambda h: np.log(1+np.exp(h))
    plt.figure(figsize=(7,3.5))
    hvals = np.linspace(-5, 5)
    plt.plot(hvals, soft(hvals),"r",markersize=3,linewidth=1)
    log = st.sidebar.slider("Select value of x", -5.0, 5.0,0.5)
    v=soft(log)
    plt.plot(log, v,'go', markersize=5)
    st.sidebar.markdown(f"## Output: {round(v,6)}")
    plt.plot([-5,5], [0,0],color="black",linewidth=0.3)
    plt.plot([0,0], [5,-0.2],color="black",linewidth=0.3)
    plt.xticks(np.arange(-5, 6, step=1))
    plt.yticks(np.arange(-0.5, 5.5, step=0.5))
    st.pyplot()
elif(selected_act=="Maxout Function"):
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/eeda24441c3129f46adeeac876c6fe3dfffb73c9',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/d714c47a3ff98f762d446f179d886ceab52ead4e',width=300)
    inputs =  np.array(st.sidebar.multiselect('Insert inputs',np.arange(1,5)))
    weights = np.array(st.sidebar.multiselect('Insert respective weights',np.arange(-5,5)))
    bias= np.array(st.sidebar.multiselect('Insert respective bias',np.arange(-2,2)))
    st.markdown(f"## input: {inputs}")
    st.markdown(f"## weights: {weights}")
    st.markdown(f"## bias: {bias}")
    if(len(inputs)==len(weights)==len(bias)!=0):
        maxout=max(inputs*weights+bias)
        st.markdown(f"## {inputs*weights+bias}")
        st.markdown(f"# Maxout output: {maxout}")
    else:
        st.sidebar.warning("Please add equal values")
    
elif(selected_act=="SoftMax Function"):
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/6d7500d980c313da83e4117da701bf7c8f1982f5',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/81a8feb8f01aaed053c103113e3b4917f936aef0',width=300)
    number = st.sidebar.multiselect('Insert a number',np.arange(-5,10))
    st.sidebar.markdown(f"## {number}")
    plt.figure(figsize=(7,5))
    softmax = lambda a: np.exp(a) / np.sum(np.exp(a))
    z=softmax(sorted(number))
    plt.plot(sorted(number), z,'ro-',linewidth=0.5,markersize=3)
    plt.plot([-5,10], [0,0],color="black",linewidth=0.3)
    plt.plot([0,0], [1,-0.1],color="black",linewidth=0.3)
    plt.xticks(np.arange(-5, 11, step=1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    st.pyplot()
    


    