# Docker를 이용한 설치 및 실행
Docker 기반 설치
- [ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)
- [Kibana](https://www.elastic.co/guide/en/kibana/current/docker.html)

<br>
<br>


## ElasticSearch
### ElasticSearch 다운로드
- ElasticSearch
    - 8.8.1 버전 다운로드

    <br>

    ```
    docker network create elastic
    ```
    
    <br>
<br>

### Docker Network 생성
- Docker의 container는 독립된 환경에서 구동되기 때문에 기본적으로는 다른 container와 통신 불가능
- 여러 container를 하나의 docker network에 연결하면 서로 통신이 가능해짐 
- ElasticSearch와 kibana container가 통신하기 위해 필요

    <br>

    ```
    docker network create elastic
    ```
    
    <br>

<br>

### ElasticSearch 구동 (single node)
 - ElasticSeach Container를 실행하기 위한 명령어

    <br>
 
    ```
    docker run --name es01 --net elastic -p 9200:9200 -it {이미지 ID}
    docker run --name es01 --net elastic -p 9200:9200 -p 9300:9300 -it {이미지 ID}

    docker ps 
    ```

    <br>

- 구동이 안되는 경우
    - ERROR: Elasticsearch exited unexpectedly 라는 에러가 발생하면서 구동이 안되는 경우 아래 명령어 실행

    <br>

    ```
    wsl -d docker-desktop
    sysctl -w vm.max_map_count=262144
    ```

<br>
<br>



#### Elastic Search의 Port 
- ElasticSearch는 cluster로 구성이되는데, 노드 하나만 실행하여도 cluster로 실행이 됨
- 그 Node는 9200번 port와 9300번 port를 통해
    - 9200번 port는 http protocol을 이용하여 client와 통신
    - 9300번 port는 tcp를 이용하여 다른 node와 통신

<br>
<br>

### 인증서 복사 및 실행
- http_ca.crt security certificate 파일을 docker host에서  local machine으로 복사
    
    <br>

    ```
    docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .
    ```

    <br>

- 새로운 터미널에서 
    <br>

    ```
    curl --cacert http_ca.crt -u elastic https://localhost:9200
    curl -k --cacert http_ca.crt -u elastic https://localhost:9200
    ```

    <br>

    - 첫 번쨰 줄에 있는 명령어를 실행했을 때 올바른 비밀번호를 입력했을 때에도 에러 발생
        - curl: (60) schannel: CertGetCertificateChain trust error CERT_TRUST_REVOCATION_STATUS_UNKNOWN
    - -k option을 추가하여 해결 

    
## kibana
### Kibana 다운로드
- Kibana image를 다운받음
    - ElasticSearch와 같은 버전으로 다운받아야 함!

    <br>
    
    ```
    docker pull docker.elastic.co/kibana/kibana:8.8.1
    ```

    <br>

<br>

### Kibana 실행
- ElasticSearch와 같은 docker network에서 container 실행

    <br>
    
    ```
    docker run --name kib-01 --net elastic -p 5601:5601 {이미지 ID}
    ```

    <br>

- 실행하고 나면 console 창에 아래와 같이 나오는데 주소를 0.0.0.0 을 localhost 바꿔 실행 

    <br>

    ```
    Go to http://0.0.0.0:5601/?code=______ to get started.
    ```

    <br>

- 위의 주소로 들어가서 enrollment token 등록
    - enrollment token이 기억이 안나는 경우 아래 명령어를 실행하여 찾음
    
    <br>    
    
    ```
    docker exec -it [es container 이름] /usr/share/elasticsearch/bin/elasticsearch-create-enrollment-token -s kibana
    ```

    <br>    

