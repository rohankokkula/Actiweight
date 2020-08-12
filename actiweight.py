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
st.markdown("""<h1 style='text-align: center;margin-top:-110px;font-family: montserrat;font-size:50px;'>ACTIWEIGHT</h1><h3 style='text-align: center;margin-top:-25px;font-family: montserrat;'>Created by: <a href='https://in.linkedin.com/in/rohankokkula'><b>Rohan Kokkula<a href="https://in.linkedin.com/in/rohankokkula" target="_blank">
  <img align="right" alt="Rohan Kokkula | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />
</a></b></a>.</h3>""", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: center;' >Select Function</h1>", unsafe_allow_html=True)
selected_act = st.sidebar.selectbox('', ("What are Activation Functions?","Sigmoid / Logistic Function","Hyperbolic Tangent Function","Rectified Linear Unit","Leaky Rectified Linear Unit","Parametric Rectified Linear Unit","Exponential Linear Unit","Swish Function","SoftMax Function","Softplus Function","Maxout Function"))
if(selected_act=="What are Activation Functions?"):
    st.image("formulae/intro.gif")
    st.markdown(f"""<h1 style='text-align: center;'>What are Activation Functions?</h1>
    <h2>Activation function is a very important feature of an artificial neural network , they basically decide whether the neuron should be activated or not.<br><br>
    These type of functions are attached to each neuron in the network, and determines whether it should be activated or not, based on whether each neuron’s input is relevant for the model's prediction.<br><br>
    Important use of any activation function is to introduce <b>non-linear</b> properties to our Network.
    <img style=' align:center;  display: block;margin-left: auto;margin-right: auto;width: 100%;' src="https://miro.medium.com/max/3000/1*T4ARzySpEQvEnr_9pc78pg.jpeg">
    In simple term , <br>it calculates a <b>"weighted sum(Wm)"</b> of its <b>input(xm)</b>, adds a <b>bias</b> and then decides whether it should be "<b>fired</b>" or not.<br><br>
    The activation function is a mathematical “gate” in between the input feeding the current neuron and its output going to the next layer. 
    <h1> Why do we use an activation function ?</h1>
    If we do not have the activation function the weights and bias would simply do a linear transformation. <br><br>
    A linear equation is simple to solve but is limited in its capacity to solve complex problems and have less power to learn complex functional mappings from data.<br><br>
    A neural network without an activation function is just a linear regression model.
    <h1>Why derivative/differentiation is used ?</h1>
    When updating the curve, to know in which direction and how much to change or update the curve depending upon the slope.<br><br>
    That is why we use differentiation in almost every part of Machine Learning and Deep Learning.</h2>""", unsafe_allow_html=True)
    st.markdown("""""", unsafe_allow_html=True)
elif(selected_act=="Sigmoid / Logistic Function"):
    st.sidebar.image('formulae/sigmoid.png',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/50a861269c68b1f1b973155fa40531d83c54c562',width=300)
    logistic = lambda h, beta: 1./(1 + np.exp(-beta * h))
    st.sidebar.info("Note: x=sum(inputs*weights) + bias")
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
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
    <h2>Sigmoid Function can be thought of as the firing rate of a neuron. In the middle where the slope is relatively large, it is the sensitive area of the neuron. On the sides where the slope is very gentle, it is the neuron's inhibitory area.<br><br>
    In the sigmoid function, we can see that its output is in the open interval (0,1). We can think of probability, but in the strict sense, don't treat it as probability.</h2>""", unsafe_allow_html=True)
    options = st.radio("",('Advantages', 'Disadvantages'))
    if(options=="Advantages"):
        st.markdown("""<li>The function is <b>differentiable</b>. That means, we can find the slope of the sigmoid curve at any two points.</li>
        <li><b>Smooth gradient</b>, preventing "jumps" in output values.</li>
        <li>Output values bound between 0 and 1, <b>normalizing</b> the output of each neuron.</li>""", unsafe_allow_html=True)
    if(options=="Disadvantages"):
        st.markdown("""<li><b>Vanishing gradient</b> - for very high or very low values of X, there is almost no change to the prediction, causing a vanishing gradient problem.</li>
        <li>Due to vanishing gradient problem, sigmoids have <b>slow convergence.</b></li>
        <li>Function output is <b>not zero-centered</b>, which will reduce the efficiency of weight update.</li>
        <li>Function performs exponential operations, which is <b>slower</b> for computers.</li>""", unsafe_allow_html=True)

elif(selected_act=="Hyperbolic Tangent Function"):
    st.sidebar.image('formulae/tanh.png',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/b371c445bf1130914d25b1995d853ac0e27bc956',width=200)
    plt.figure(figsize=(7,3.5))
    hyperbolic_tangent = lambda h: (np.exp(h) - np.exp(-h)) / (np.exp(h) + np.exp(-h))
    theta = st.sidebar.slider("Select value of θ", -1, 25,1)
    hvals = np.linspace(-5, 5)
    plt.plot(hvals, hyperbolic_tangent(hvals*theta),"r",markersize=3,linewidth=1)
    st.sidebar.info("Note: x=sum(inputs*weights) + bias")
    tan = st.sidebar.slider("Select value of x", -5, 5,1)
    v=hyperbolic_tangent(tan*theta)
    plt.plot(tan, v,'go', markersize=5)
    st.sidebar.markdown(f"## Output: {round(v,6)}")
    plt.xticks(np.arange(-5, 6, step=1))
    plt.yticks(np.arange(-1, 1.2, step=0.5))
    plt.plot([-5,5], [0,0],color="black",linewidth=0.3)
    plt.plot([0,0], [-1,1],color="black",linewidth=0.3)
    st.pyplot()
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
    <h2>The curves of tanh function and sigmod function are relatively similar. <br>First of all, when the input is large or small, the output is almost smooth and the gradient is small, which is not conducive to weight update. The difference is the output interval.<br><br>
    In general binary classification problems, the tanh function is used for the hidden layer and the sigmod function is used for the output layer.</h2>""", unsafe_allow_html=True)
    options = st.radio("",('Advantages', 'Disadvantages'))
    if(options=="Advantages"):
        st.markdown("""<li><b>Zero-centered</b> — making it easier to model inputs that have strongly negative, neutral, and strongly positive values.</li>
        <li><b>Smooth gradient</b>, preventing "jumps" in output values.</li>
        <li>Output values bound between -1 and 1, <b>normalizing</b> the output of each neuron.</li>""", unsafe_allow_html=True)
    if(options=="Disadvantages"):
        st.markdown("""<li><b>Vanishing gradient</b> - for very high or very low values of X, there is almost no change to the prediction, causing a vanishing gradient problem.</li>
        <li>Due to vanishing gradient problem, sigmoids have <b>slow convergence.</b></li>
        <li>Function output is <b>not zero-centered</b>, which will reduce the efficiency of weight update.</li>
        <li>Function performs exponential operations, which is <b>slower</b> for computers.</li>""", unsafe_allow_html=True)

elif(selected_act=="Rectified Linear Unit"):
    st.sidebar.image('formulae/relu.png',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/e8723cef7eb5dedf4aa20e174ee281b76a6cbec4',width=300)
    st.sidebar.info("Note: x=sum(inputs*weights) + bias")
    x = np.linspace(0, 10)
    plt.figure(figsize=(5,3))
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
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
                <h2>The rectified linear activation function is a piecewise linear function that will output the input directly if is positive, otherwise, it will output zero.<br><br>
                It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.</h2>""", unsafe_allow_html=True)
    options = st.radio("",('Advantages', 'Disadvantages'))
    st.markdown(f"## {options}")
    if(options=="Advantages"):
        st.markdown("""<li>The function and its derivative both are <b>monotonic.</b></li>
                    <li>When the input is positive, there is no gradient saturation problem.</li>
        <li><b>Computationally efficient</b>—allows the network to converge very quickly.</li>
        <li><b>Non-linear</b>—although it looks like a linear function, ReLU has a derivative function and allows for backpropagation</li>""", unsafe_allow_html=True)
    if(options=="Disadvantages"):
        st.markdown("""<li><b>The Dying ReLU problem</b> — when inputs approach zero, or are negative, the gradient of the function becomes zero, the network cannot perform back-propagation and cannot learn.</li>
        <li>We find that the output of the ReLU function is either 0 or a positive number, which means that the ReLU function is <b>not a zero-centric function.</b></li>""", unsafe_allow_html=True)


elif(selected_act=="Leaky Rectified Linear Unit"):
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/aaabce8985d074b5f4482f4efa327c7c61da3ca6',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/53ae28dca48ccda3b6ac4d5b0ecec23b70368784',width=300)
    st.sidebar.info("Note: x=sum(inputs*weights) + bias")
    x = np.linspace(0, 10)
    plt.figure(figsize=(7,4))
    plt.plot(x, x + 0, linestyle='solid')
    plt.plot([-100,10], [0,0],color="black",linewidth=0.5)
    plt.plot([0,0], [-1,10],color="black",linewidth=0.5)
    y = st.sidebar.slider("Select value of x", -100, 10,-62)
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
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
                <h2>In order to solve the Dying ReLU Problem, people proposed to set the first half of ReLU 0.01x instead of 0. <br><br>
                The leak helps to increase the range of the ReLU function.</h2>""", unsafe_allow_html=True)
    options = st.radio("",('Advantages', 'Disadvantages'))
    if(options=="Advantages"):
        st.markdown("""<li><b>Prevents dying ReLU problem</b>— this variation of ReLU has a small positive slope in the negative area, so it does enable back-propagation, even for negative input values.</li>""", unsafe_allow_html=True)
    if(options=="Disadvantages"):
        st.markdown("""<li><b>Inconsistent Results</b> — leaky ReLU does not provide consistent predictions for negative input values.</li>
                    <li>During the front propagation if the learning rate is set very high it will overshoot killing the neuron.</li>""", unsafe_allow_html=True)

elif(selected_act=="Parametric Rectified Linear Unit"):
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/387a2af979ccc6a29b62950e1efb7c3a86209ad7',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/51480bf026d4e3149f7c815fda04940663894791',width=300)
    plt.figure(figsize=(7,4))
    y = st.sidebar.slider("Select value of x", -100, 10,-20)
    plt.plot([-60,30], [0,0],color="black",linewidth=0.5)
    plt.xticks(np.arange(-100,11, step=10))
    if(y<0):
        alpha = st.sidebar.slider("Select value of Alpha", 0.01, 1.0,0.4,step=0.1)
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
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
                <h2>The idea of leaky ReLU can be extended even further. Instead of multiplying x with a constant term we can multiply it with a hyper-parameter which seems to work better the leaky ReLU.<br><br>
                 Inshort, replace 0.01 with &alpha; which could be the learnable parameter for backpropagation.<br><br>
                In the negative region, PReLU has a small slope, which can also avoid the problem of ReLU death.<br> Compared to ELU, PReLU is a linear operation in the negative region. <br><br>Although the slope is small, it does not tend to 0, which is a certain advantage.</h2>""", unsafe_allow_html=True)
    options = st.radio("",('Advantages', 'Disadvantages'))
    if(options=="Advantages"):
        st.markdown("""<li>Allows the negative slope to be learned — unlike leaky ReLU, this function provides the slope of the negative part of the function as an argument.<br> It is, therefore, possible to perform backpropagation and learn the most appropriate value of α.</li>""", unsafe_allow_html=True)
    if(options=="Disadvantages"):
        st.markdown("""<li>May perform differently for different problems.</li>
                    <li>Same as Leaky ReLU</li>""", unsafe_allow_html=True)

elif(selected_act=="Exponential Linear Unit"):
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/5d740c6ed2015b0208a6945e53f10d89c11855b3',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/ad31fcdb29df53296c148a3af5c1b66bc7317a90',width=300)
    plt.figure(figsize=(7,4))
    y = st.sidebar.slider("Input Value", -100, 10,-15)
    plt.plot([-60,30], [0,0],color="black",linewidth=0.5)
    if(y<0):
        alpha = st.sidebar.slider("Alpha Value", 1, 5,2)
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
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
                <h2>ELU is proposed to solve the problems of ReLU. Due to exponent nature for negative inputs, the multiplier(&alpha;) is not seen in this case.</h2>""", unsafe_allow_html=True)
    options = st.radio("",('Advantages', 'Disadvantages'))
    if(options=="Advantages"):
        st.markdown("""<li>No Dead ReLU issues.</li>
                    <li>The mean of the output is close to 0, zero-centered.</li>""", unsafe_allow_html=True)
    if(options=="Disadvantages"):
        st.markdown("""<li>One small problem is that it is slightly more computationally intensive for negative values.</li>
                    <li>Same as Leaky ReLU</li>""", unsafe_allow_html=True)

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
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
                <h2>The formula is: y = x * sigmoid(x)<br><br>
                Swish is a new, self-gated activation function discovered by researchers at Google.<br><br>
                Swish's design was inspired by the use of sigmoid functions for gating in LSTMs and highway networks.<br><br> We use the same value for gating to simplify the gating mechanism, which is called self-gating.</h2>""", unsafe_allow_html=True)
    options = st.radio("",('Advantages', 'Disadvantages'))
    if(options=="Advantages"):
        st.markdown("""<li>It only requires a simple scalar input, while normal gating requires multiple scalar inputs. This replaces activation functions that take a single scalar as input (such as ReLU) without changing the hidden capacity or number of parameters.</li>
                    <li>Smoother version of ReLU.</li>""", unsafe_allow_html=True)
    if(options=="Disadvantages"):
        st.markdown("""<li>One small problem is that it is slightly more computationally intensive for negative values.</li>
                    <li>Same as ReLU</li>""", unsafe_allow_html=True)

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
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
                <h2>The softplus function is similar to the ReLU function, but it is relatively smooth. It is unilateral suppression like ReLU.<br><br></h2>""", unsafe_allow_html=True)
    options = st.radio("",('Advantages', 'Disadvantages'))
    if(options=="Advantages"):
        st.markdown("""<li>It only requires a simple scalar input, while normal gating requires multiple scalar inputs. This replaces activation functions that take a single scalar as input (such as ReLU) without changing the hidden capacity or number of parameters.</li>
                    <li>Smoother version of ReLU.</li>""", unsafe_allow_html=True)
    if(options=="Disadvantages"):
        st.markdown("""<li>One small problem is that it is slightly more computationally intensive for negative values.</li>
                    <li>Same as ReLU</li>""", unsafe_allow_html=True)

elif(selected_act=="Maxout Function"):
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/eeda24441c3129f46adeeac876c6fe3dfffb73c9',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/d714c47a3ff98f762d446f179d886ceab52ead4e',width=300)
    inputs =  np.array(st.sidebar.multiselect('Insert inputs',np.arange(1,5)))
    weights = np.array(st.sidebar.multiselect('Insert respective weights',np.arange(-5,5)))
    bias= np.array(st.sidebar.selectbox('Insert respective bias',np.arange(-2,2)))
    st.markdown(f"## input: {inputs}")
    st.markdown(f"## weights: {weights}")
    st.markdown(f"## bias: {bias}")
    if(len(inputs)==len(weights)!=0):
        maxout=max(inputs*weights+bias)
        st.markdown(f"## {np.dot(inputs,weights)+bias}")
        st.markdown(f"# Maxout output: {maxout}")
    else:
        st.sidebar.warning("Please add equal values")
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
                <h2>The Maxout activation is a generalization of the ReLU and the leaky ReLU functions.<br> It is a learnable activation function.<br><br>
                Notice that both ReLU and Leaky ReLU are a special case of this form (for example, for ReLU we have w1,b1 =0).<br><br>The Maxout neuron therefore enjoys all the benefits of a ReLU unit (linear regime of operation, no saturation) and does not have its drawbacks.<br><br>
                Maxout can be seen as adding a layer of activation function to the deep learning network, which contains a parameter k.<br><br> Compared with ReLU, sigmoid, etc., this layer is special in that it adds k neurons and then outputs the largest activation value.<br><br> We use the same value for gating to simplify the gating mechanism, which is called self-gating.</h2>""", unsafe_allow_html=True)

elif(selected_act=="SoftMax Function"):
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/6d7500d980c313da83e4117da701bf7c8f1982f5',width=300)
    st.sidebar.image('https://wikimedia.org/api/rest_v1/media/math/render/svg/81a8feb8f01aaed053c103113e3b4917f936aef0',width=300)
    st.sidebar.info("Note: x=sum(inputs*weights) + bias")
    number = st.sidebar.multiselect('Insert value of x',np.arange(-5,10))
    st.sidebar.markdown(f"#### Input to SoftMax Function")
    st.sidebar.markdown(f"# {number}")
    plt.figure(figsize=(7,4))
    softmax = lambda a: np.exp(a) / np.sum(np.exp(a))
    z=softmax(sorted(number))
    plt.plot(sorted(number), z,'bo',linewidth=0.5,markersize=5)
    plt.plot([-5,10], [0,0],color="black",linewidth=0.3)
    plt.plot([0,0], [1,-0.1],color="black",linewidth=0.3)
    plt.xticks(np.arange(-5, 11, step=1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    st.pyplot()
    st.markdown(f"""<h1 style='text-align: center;'>{selected_act}</h1>
                <h2>Softmax function calculates the probabilities distribution of the event over 'n' different events.<br><br>
                It takes as input a vector z of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers. <br><br>
                That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1;<br> but after applying softmax, each component will be in the interval (0,1) and the components will add up to 1, so that they can be interpreted as probabilities.<br><br>
                Furthermore, the larger input components will correspond to larger probabilities.</h2>""", unsafe_allow_html=True)

