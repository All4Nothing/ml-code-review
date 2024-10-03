**[Generative Adversarial Network paper에 대해 정리한 글](https://github.com/All4Nothing/papers-repo/tree/main/Generative%20Adversarial%20Nets)**

## Generative Adversarial Network

**Discriminative Model**: learns how to classify input to its class
ex) Input Image (64x64x3) → **Discriminative Model** → man
**→** **Supervised Learning** 

**Generative Model** : learns the distribution of traing data ex) Latent code(100) → **Generative Model** → Image(64x64x3) 
**→** **Unsupervised Learning** 

The goal of the generative model is to find a $p_{model}(x)$ that approximate $p_{data}(x)$ well.

두 개의 확률분포의 차이를 줄여주는게 generative model의 목표값이 된다.

## GAN
![gan](https://github.com/All4Nothing/GAN/assets/81239098/81be4662-61f2-4aeb-b873-f7fa11f0f927)


$x$ → **D** → $D(x)$

$z$ → **G** → $G(z)$ → **D** → $D(G(z))$

Discriminator($D(x)$)는 진짜와 가짜 image를 구분

Generator($G(z)$는 랜덤한 코드(sample latent code $z$ from Gaussian distiribution)를 가지고 가짜 이미지를 만든다.

### Equation

$\underset{G}{min}\ \underset{D}{max}\ V(D,G) = E_{x∼p_{data(x)}}[logD(x)]+E_{z∼p_z(z)}[log(1−D(G(z)))]$

- $x$ : Sample x from real data distribution
- $z$ : Sample latent code $z$ from Gaussian distribution

**Discriminator**

- $D$ should maximize $V(D,G)$
- $E_{x∼p_{data(x)}}[logD(x)]$ : Maximum when $D(x)$ = 1
- $E_{z∼p_z(z)}[log(1−D(G(z)))]$ : Maximum when $D(G(z))$  = 0

**Generator**

- $G$ should minimize $V(D,G)$
- $E_{x∼p_{data(x)}}[logD(x)]$ : $G$ is independent of this part
- $E_{z∼p_z(z)}[log(1−D(G(z)))]$ : Minimum when $D(G(z))$=1

### Code(단순 설명용)

```python
import torch
import torch.nn as nn

D = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

D = nn.Sequential(
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Tanh()
)

criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.01)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.01)

# Assume x be real images of shape (batch_size, 784)
# Assume z be random noise of shape (batch_size, 100)

while True:
    # train D
    loss = criterion(D(x), 1) + criterion(D(G(z)), 0)
    loss.backward()
    d_optimizer.step()

    # train G
    loss = criterion(D(G(z)), 1)
    loss.backward()
    g_optimizer.step() 
    
    ## Adam에 G.parameters()를 넣었기에 G를 학습할 때 D의 parameters는 update되지 않음
```

### Variants of GAN

- Deep Convolutional Generative Adversarial Network (DCGAN)
- Least Squares GAN (LSGAN)
    - 기존의 GAN은 Discriminator를 속이기만 하면 됐음.
    - Discriminator를 속여도, 실제 image data와 먼 image data를 Generator가 생성하면 의미가 없음
    - Discriminator에서 sigmod를 빼고, cross-entropy loss → L2 loss로 변경
- Semi-Supervised GAN (SGAN)
- Auxiliary Classifier GAN (ACGAN)
    - SGAN은 가짜 이미지에 fake label(가짜 2 이미지는 fake라고)를 붙임. ACGAN은 가짜 이미지도 label을 붙임(가짜 2 이미지도 fake라고 구분하면서도, 2라고 label을 붙임)

### Extensions of GAN

- CycleGAN
    - Unpaired Image-to-Image Translation
    - presents a GAN model that transfer an image from a source domain A to a target domain B in the absence of paired examples.
    - 얼룩말 이미지를 말 이미지로 바꿔서, 진짜 말과 가짜 말을을 구분하는 Discrminator를 속인다면, 굳이 얼룩말 사진과 비슷한 형태가 아닌 전혀 다른 말 이미지를 생성해도 속일 수 있음. 이를 막기 위해, 얼룩말 이미지로 가짜 말 이미지를 만들고($G_{AB}$), 다시 가짜 말 이미지로 얼룩말 이미지를 만드는 Generator($G_{BA}$)를 이용
- StackGAN : Text to Photo-realistic Image Synthesis
    - 한번에 고해상도 이미지를 generate하기 보다, 저해상도 이미지를 생성하는 generator와 화질을 upscaling하는 generator 2개 이용
    - 저해상도 이미지를 fake or real 구분하는 discriminator와 고해상도 이미지를 fake or real 구분하는 discriminator 2개 존재.

### 참고한 Reference

- [1시간만에 GAN(Generative Adversarial Network) 완전 정복하기](https://www.youtube.com/watch?v=odpjk7_tGY0&t=323s)
- [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/generative_adversarial_network)
