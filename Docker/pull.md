


## Docker Hub
- 프로그램에 필요한 것이 있으면 app store와 같은 곳에서 찾는 것과 같이
Docker hub 라고 부르는 registry라고 하는 서비스에서 필요한 software 찾게 됨
- App store에서 다운로드 받아서 PC에 저장하면 program이라 하고 Docker hub에서 필요한 것을 다운로드해서 가지고 있는 것을 image 라고 함
- Program을 실행하면 process라는 것이 실제로 동작하는 것과 같이 image를 실행하는 것을 container라고 함
- Program이 여러개의 process를 가질 수 있는 것처럼 image도 여러개의 container를 가질 수 있음

- Docker container에서 image를 다운받는 것을 pull 이라고 하고 image를 실행시키는 행위를 run 이라고 함
- Run을 하게 되면 image가 container가 되고, 그 container 실행되면서 그 container 안에 포함되어있는 실행되도록 조치되어있는 프로그램이 실행

<br>

## Image 다운
[공식 홈페이지](https://docs.docker.com/engine/reference/commandline/pull/)
1. https://hub.docker.com/ 에 들어가서 Explore 버튼 클릭 
2. Containers 클릭
    - 현재는 없는 듯
3. 명령어 확인 
4. 

### Example
1. docker image 다운로드
    
    <br>

    ```
    docker pull httpd 
    ```

<br>

2. image 다운 확인
    
    <br>

    ```
    docker images
    ```

<br>
<br>

## Container 실행 : run

1. Container 생성 및 실행
    
    <br>
    
    ```
    docker run [OPTIONS] IMAGE [COMMAND] [ARGS]
    docker run httpd
    docker run --name ws2 httpd 
    docker run --name ws3 -p 8081:80 httpd
    ```

    <br>

    - 하나의 이미지는 여러개의 container 만드는 것 가능
    - --name 이라는 OPTIONS 은 container의 이름 지정 가능
    - -p 라는 OPTIONS은 container의 local host와 container의 port 설정 가능

<br>
<br>


2. Container 확인

    <br>

    ```
    docker ps 
    docker ps -a
    ```

    <br>

    - Container에 대한 정보 알 수 있음
        - 실행중이지 않고 멈춰있는 container까지 보고싶다면 -a

<br>
<br>

 

3. 실행 중인 container 중지

    <br>

    ```
    docker stop CONTAINER
    docker stop ws2
    ```
    
    <br>

    - container의 이름이나 ID 를 넣어서 멈춤

<br>
<br>

4. 중지시킨 container 재실행

    <br>

    ```
    docker start CONTAINER
    docker start ws2
    ```
    
    <br>

<br>

5. Container log 확인하는 방법

    <br>

    ```
    docker logs [OPTIONS] CONTAINER
    docker logs ws2
    docker logs -f ws2
    ```
    
    <br>

    - OPTIONS 으로 -f 가 주어지면 실시간으로 watching 하게 됨 

<br>
<br>

6. Container 삭제

    <br>

    ```
    docker rm [OPTIONS] CONTAINER [CONTAINER ...]
    docker rm ws2
    docker rm --force ws2
    ```
    
    <br>

    - 현재 실행중인 container는 지울 수 없기 때문에  `docker stop` 명령어로 container 종로 후 docker rm 실행
    - 또는 --force 라는 OPTIONS 을 사용하여 stop 하지 않고 바로 삭제 가능

<br>
<br>

7. Image 삭제 

    <br>

    ```
    docker rmi [OPTIONS] IMAGE [IMAGAE ...]
    docker rmi httpd
    ```
    
    <br>

<br>
<br>



- 실행 중인 Container에 명령 사용

    <br>

    ```
    docker exec [OPTIONS] CONTAINER COMMAND [ARG ...]
    docker exec ws3 pwd
    docker exec ws3 ls

    docker exec -it ws3 /bin/sh
    docker exec -it ws3 /bin/bash
    docker exec -itu0 ws3 /bin/bash

    cd /usr/local/apache2/htdocs/
    apt update
    apt install nano
    nano index.html

    `i`` 

    <br>

    - exec 명령어 뒤에 container 이름 써 해당 container에 명령어 사용 가능
        - 앞부분을 계속 사용하는 것이 번거로움
    - /bin/sh, /bin/bash라는 명령어를 수행하게 되면 
        - shell 프로그램 실행
            - 사용자가 입력한 명령을 shell 프로그램이 받아 운영체제가 전달 
        - -i 라는 OPTIONS 을 주게 되면 지속적으로 연결이 됨 
        - -it 를 쓰면 terminal 과 container의 지속적인 연결이 가능하게 됨
        - exit 를 쓰면 다시 host를 향해 주는 명령어
    - 권한 문제가 발생하면 -itu0 를 사용