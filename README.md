# 原神小草神AI聊天虚拟人  
## 简介   
🤖 纳西妲风格虚拟人。该对话系统语言风格参考对象为原神角色纳西妲，世界观来源为原神中纳西妲角色简介和任务。   
📕 预训练和用于RAG的数据来源于B站原神wiki，来源网站：[原神B站wiki](https://wiki.biligame.com/ys/%E9%A6%96%E9%A1%B5)。SFT源数据也来源于B站原神wiki，并通过AI生成、人工矫正构建成数据集。  
🔆 默认Chat模型为Lora微调后的Qwen1.5-1.8B和Qwen1.5-4B模型，也可以使用GPT系列(需要api key，在[config](config.ini)中添加gpt_api_key信息)，默认Embedding模型为[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)，默认Rerank模型为[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)。  
## 与纳西妲对话
角色设定  
![本地图片](pics/人物设定.png "相对路径演示,上一级目录")
角色关系  
![本地图片](pics/人物关系.png "相对路径演示,上一级目录")  
认知观念  
![本地图片](pics/认知观念.png "相对路径演示,上一级目录")  
道德准则  
![本地图片](pics/道德准则.png "相对路径演示,上一级目录")
## 🏎️开始  
进入想要存放项目的目录，执行以下命令：
```angular2html
git clone https://github.com/ZeroZY-bgp/pocket_nahida.git
```
进入pocket_nahida目录，创建虚拟环境：
```angular2html
create -n venv python=3.9
pip install -r requirements.txt
```
安装完成后，运行webui.py文件：
```angular2html
python webui.py
```
💡 第一次运行系统会自动下载所需要的模型，模型下载默认从[huggingface](https://huggingface.co/)中下载，需要等待一段时间。模型默认下载路径在[config](config.ini)的model_cache_dir中
## 💻需求  
文件操作支持Windows系统和Ubuntu系统，建议在Ubuntu服务器下使用。  
建议环境python版本为3.9及以上。
- 模型硬件要求：  
为了推理速度，建议使用GPU，显存至少为12G。  
默认使用的Chat模型为Lora微调后的Qwen1.5-1.8B和Qwen1.5-4B模型，约占显存8G。  
默认使用的Embedding模型为BAAI/bge-small-zh-v1.5约占显存2G，可修改在CPU中运行。  
默认使用的Rerank模型为BAAI/bge-reranker-v2-m3约占显存2G，可修改在CPU中运行。   
## 🛠️高级  
[config](config.ini)为基础配置。如果你的配置不够可以将model_quantized改为True量化加载chat模型，但是对话效果会不理想。  
如果你有更好的chat模型，可以将model_name_or_path改为你的模型路径。

