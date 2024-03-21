data = """  
alabaster=0.7.12=py36_0  
anaconda=2019.03=py36_0  
anaconda-client=1.7.2=py36_0  
anaconda-project=0.8.2=py36_0  
asn1crypto=0.24.0=py36_0  
astroid=2.2.5=py36_0  
astropy=3.1.2=py36h7b6447c_0  
atomicwrites=1.3.0=py36_1  
attrs=19.1.0=py36_1  
babel=2.6.0=py36_0  
backcall=0.1.0=py36_0  
backports=1.0=py36_1  
backports.os=0.1.1=py36_0  
backports.shutil_get_terminal_size=1.0.0=py36_2  
beautifulsoup4=4.7.1=py36_1  
bitarray=0.8.3=py36h14c3975_0  
bkcharts=0.2=py36h735825a_0  
blas=1.0=mkl  
blosc=1.15.0=hd408876_0  
bokeh=1.0.4=py36_0  
boto=2.49.0=py36_0  
bottleneck=1.2.1=py36h035aef0_1  
bzip2=1.0.6=h14c3975_5  
ca-certificates=2019.1.23=0  
cairo=1.14.12=h8948797_3  
certifi=2019.3.9=py36_0  
cffi=1.12.2=py36h2e261b9_1  
chardet=3.0.4=py36_1  
click=7.0=py36_0  
cloudpickle=0.8.0=py36_0  
clyent=1.2.2=py36_1  
colorama=0.4.1=py36_0  
contextlib2=0.5.5=py36h6c84a62_0  
cryptography=2.6.1=py36h1ba5d50_0  
cudatoolkit=9.0=h13b8566_0  
curl=7.64.0=hbc83047_2  
cycler=0.10.0=py36h93f1223_0  
cython=0.29.6=py36he6710b0_0  
cytoolz=0.9.0.1=py36h14c3975_1  
"""

data = data.replace("=", "")  # 去除每个 "=" 后面的所有内容
data = data.strip()  # 去除每个 "=" 前面的空格和换行符

print(data)