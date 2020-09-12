# HMM_HWC
------------------------------------------------------------------------
### BORDINO ALBERTO <br>
### HIDDEN MARKOV MODEL: CONCETTI, MODELLI E APPLICAZIONI
-----------------------------------------------------------------------
-----------------------------------------------------------------------
Si vogliono applicare gli HMM al problema di riconoscimento testuale. <br>
Il dataset è disponibile a: www.kaggle.com/dataset/febc0a2705c938eef548e93184b2e23bc775bf1aff1bd03f12278c32e78dbe93 <br>
Si tratta del MNIST dataset che contiene 70000 immagini di cifre scritte a mano: 60000 per il training, 10000 per il testing. 
Ogni riga rappresenta un'immagine: la prima colonna è il label (0-9), le successive 784 rappresentano il valore in scala di grigi (0-255) di ciascuno dei 28x28 pixel che costituiscono l'immagine. <br>

![Image of Mnist](https://github.com/abordino/HMM_HWC/blob/master/mnist.png)

Si allena un modello per ogni cifra con l'algoritmo di Baum-Welch e lo si testa, confrontando l'accuratezza ottenuta con altri metodi di classificazione: http://yann.lecun.com/exdb/mnist/

Si implementa infine una GUI con cui l'utente possa disegnare una cifra e chiedere al modello di classificarla.
