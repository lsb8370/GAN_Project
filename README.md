# GAN_Project
 Game AI 수업 기말과제로 CycleGAN을 사용하여 프로젝트를 진행함
 
### 프로젝트 개요
 
#### 프로젝트 동기
    GAN을 사용한 사례들을 찾아보던 중 사진을 특정 화가의 화풍으로 바꿔주는 것이 눈길을 끌었다.
    
    그렇다면 특정 사물의 사진을 넣으면 Line-Drawing 형태로 바꿔줄 수는 없을까 생각하게 되었고 그 중 비행기를 선택하게 되었다.
    
    비행기를 선택한 이유는 비행기마다 겉에 붙어있는 데칼도 다르고 활주로에 있는 모습과 하늘에 있는 모습이 있을텐데 이런 요소들이
    
    과연 학습에 영향을 미칠지 궁금했기 때문이다.
    
    
#### 프로젝트 목표

    비행기 사진을 Input으로 넣으면 라인드로잉 형태로 Output이 나옴
    

#### 프로젝트 진행 순서

    1) 데이터셋 찾기

    2) 프로젝트에 사용하기에 적당한 신경망 선정하기

    3) 학습 진행
    
    
#### 프로젝트에 필요한 데이터셋

    * 비행기 사진
    
    * 비행기를 라인 드로잉 형태로 그린 이미지
    
    
### 프로젝트 진행

#### 선정된 신경망

    CycleGAN
    
#### 선정된 이유 

   - 조사한 결과 보통 이런 프로젝트는 `pix2pix`나 `CycleGAN`을 사용함
    
   - `pix2pix`는 Discriminator 와 Generator에 `paired data`를 넣어줌
    
      `CycleGAN`은 `unpaired data`를 넣어줌
    
     ➜ paired data는 데이터를 수집하는데 시간이 많이 걸림
       
       만약, pair가 되는 데이터가 없다면 직접 만들어야 함
       
       예 ) 비행기 사진은 있는데 그 사진을 라인드로잉으로 그린 이미지는 없다면 직접 그려야 함
   
     ➜ pix2pix의 이러한 단점때문에 `CycleGAN` 선정함
     
#### 사용할 데이터셋

1] 비행기 사진
   
* 논문 : Fine-Grained Visual Classification of Aircraft, S. Maji, J. Kannala, E. Rahtu, M. Blaschko, A. Vedaldi, arXiv.org, 2013

  http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/#ack
  
  
2] 비행기 Line Drawing 이미지
  
* 논문 : The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies

  http://sketchy.eye.gatech.edu/
    
    
    
