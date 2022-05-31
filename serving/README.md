# Serving

* Makefile의 8번째 줄에 port번호를 해당 서버의 포트번호로 바꾸면 서버에서 사용가능하다.
* 터미널에서 serving 디렉토리 내에서 'make -j 2 run_app'를 입력하면 사용할 수 있다.
* lib 디렉토리는 YOLOP에서 사용하는 코드들이 들어있는 디렉토리이다.
* 서빙에서 있어서는 app/model.py, app/fronted.py, app/main.py만 수정하면 된다.
* 다른 서버와 같이 사용할 때 request.post할 때 해당 서버의 주소로 넣어주면 같이 사용할 수 있을 듯...? 잘모르겠다.
* pth 파일경로는 model.py에 get_model에서 수정해줘야한다.
