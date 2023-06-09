## Install
### Install for windows
- Windows에 docker를 설치하기 전에 WSL 설치가 우선
https://velog.io/@woody_/Docker-%EC%84%A4%EC%B9%98Window-11


## Network 
### 
- 2 대의 컴퓨터가 존재
    - 한 대에는 web browser 설치
    - 한 대에는 web server 설치
- Web server에서 File System이 있고 특정 디렉토리에 파일이 존재하고 이를 webserver에게 설정
- Port
    - 여러 Software를 network적으로 구분
    - 없다면 server에 접속했을 때 어떤 sw를 사용해야 할 지 알 수 없음
- Webserver는 80번 port에서 대기하고 있도록 설정되어 있음
- webserver가 설치되어있는 주소는 test.com

- http://test.com:80/index.html 라는 주소를 주소창에 입력하면 web browser는 test.com의 80번 포트에 접속
- 80번 port에는 web server가 대기중이기 때문에 web server로 요청이 전달 
- web server는 /usr/local/apache2/htdocs/에서 index.html 파일을 찾음
- index.html 코드를 읽어서 web browser에게 index.html 코드를 전달하면 이 과정이 끝남

### Docker 이용 
- Web server가 container에 설치
- 이 Container가 설치된 운영체제는 docker host 라고 함
    - 하나의 docker host에는 여러 개의 container 존재 가능
- Docker와 container 모두 독립적인 실행 환경이기 때문에 각자 독립적인 port와 file system을 가짐
- `$ docker run httpd` 이렇게 연결하면 안됨
    - host와 container의 연결이 끊겨있기 때문
    - host의 80번 port와 container의 80번 port를 연결해주어야함
- `$ docker run httpd -p 80:80 httpd`
    - 앞의 80은 host의 port, 뒤의 80은 container의 port




### Container, Host 파일 시스템 연결 
- Container를 사용하는 이유는 필요할 때 언제든지 생성한 후 사용할 수 있다는 장점이 있기 때문
- 하지만 container의 file system 안에 파일을 저장을 하면 container가 사라짐과 동시에 파일도 사라지게 됨
- Host file system의 /desktop/htdocs와 Container의 file system 디렉토리를 연결하고 host에서 수정을 했을 때, Container의 파일 시스템에 반영되도록

```
docker run -p 8888:80 -v /test/htdocs:/usr/local/apach2/htdocs httpd
```