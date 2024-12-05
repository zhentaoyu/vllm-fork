from redisvl.index import SearchIndex
from redis import Redis

from redis.commands.search.query import Query
import subprocess
import requests
from tqdm import tqdm
import math
import logging
import os

log_level = os.getenv('LOGLEVEL', 'INFO')
logging.basicConfig(level=log_level.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

def get_service_cluster_ip(service_name):
    try:
        # Run the kubectl command to get the services
        result = subprocess.run(["kubectl", "get", "svc"], capture_output=True, text=True, check=True)

        # Parse the output
        lines = result.stdout.splitlines()
        headers = lines[0].split()

        # Find the indices for the columns we are interested in
        name_idx = headers.index("NAME")
        cluster_ip_idx = headers.index("CLUSTER-IP")
        port_idx = headers.index("PORT(S)")

        for line in lines[1:]:
            columns = line.split()
            if columns[name_idx] == service_name:
                cluster_ip = columns[cluster_ip_idx]
                ports = columns[port_idx]

                main_part = ports.split("/")[0]
                port = main_part.split(":")[0]
                return cluster_ip, port

        raise ValueError(f"Service {service_name} not found.")

    except subprocess.CalledProcessError as e:
        print(f"Error running kubectl command: {e}")
        return None


def get_template(mode="en"):
    template = str()
    if mode == "zh":
        template = """你是一位新闻编辑，现在，你被提供了1个问题，和根据这些问题检索到的文档，请分别检索内容和你自身的知识回答这些问题。以下是个例子：

问题：上海和成都市体育局在促进体育消费和全民健身运动方面有哪些相似和不同的措施？

检索文档: 在第15个全民健身日来临之际，上海市体育局将联合美团、大众点评发放500万元体育消费券，3000多家上海本地运动门店参与其中，共同点燃全民健身运动热情，促进体育消费增长。▲8月5日上午10点，上海市体育局将联合美团、大众点评发放新一轮体育消费券2023年上海体育消费券以“全民优惠健身，共享美好生活”为主题，在8月5日-9月3日期间分四期进行发放。第一期消费券发放时间为8月5日10：00-8月13日24：00，第二期消费券发放时间为8月14日-8月20日，第三期8月21日-8月27日，第四期8月28日-9月3日。实时定位在上海的消费者，可以在发放时间内进入美团、大众点评App，搜索“上海体育消费券”进行领取。为满足消费者更多个性化的需求，本轮体育消费券活动准备了满200减80、满120减50、满60减30、满40减20、满20减10和满10减5共六个面额的消费券，消费者可按需领用，先到先得。每位消费者每期最多可领取3张消费券，且每位消费者同时最多可持有3张。据“上海体育”公众号介绍，本次体育消费券适用场景多、覆盖范围广、优惠力度大。在发布会上，成都市体育局副局长陈志介绍，以成都大运会筹办举办为契机，成都积极开展“爱成都·迎大运”“运动成都·悦动生活”“万千商家齐参与”等主题体育消费促进活动，发放各类体育消费券和惠民运动券，促进体育消费持续稳步增长。2022年成都体育消费总规模为578.6亿元，居民人均体育消费为2720.6元。      ▲8月4日，成都大运会体操项目女子个人全能决赛看台上，观众为比赛队员加油 资料配图 摄影 陶轲  为持续激发体育消费活力和增长潜力，下一步，成都将持续深化体育消费试点工作，积极推进体育消费提质扩容。启动户外运动季活动，发布十大最受欢迎时尚运动消费场景。  具体而言，陈志介绍说，成都将加快推动“体育＋会展＋消费”平台建设，办好中国（成都）生活体育大会、“巴山蜀水·运动川渝”体育旅游休闲消费季、世界赛事名城发展大会、中国国际体育用品博览会等重大体育展会活动，为城市体育消费增长提供更多资源链接。

回答：上海市体育局联合美团、大众点评发放了总额500万元的体育消费券，覆盖3000多家本地运动门店，并设置了不同面额的消费券供消费者领取。而成都市体育局则是利用成都大运会的契机发放各类体育消费券和惠民运动券，同时计划通过举办大型体育展会活动和推动“体育＋会展＋消费”平台建设来进一步促进体育消费的提质扩容。

### Search results: {context}

### Question:"""

    if mode == "zh_test":
        template = """你是一位新闻编辑，现在，你被提供了1个问题，和根据这些问题检索到的文档，请分别检索内容和你自身的知识回答这些问题。以下是个例子：

问题：上海和成都市体育局在促进体育消费和全民健身运动方面有哪些相似和不同的措施？

检索文档: 在第15个全民健身日来临之际，上海市体育局将联合美团、大众点评发放500万元体育消费券，3000多家上海本地运动门店参与其中，共同点燃全民健身运动热情，促进体育消费增长。▲8月5日上午10点，上海市体育局将联合美团、大众点评发放新一轮体育消费券2023年上海体育消费券以“全民优惠健身，共享美好生活”为主题，在8月5日-9月3日期间分四期进行发放。第一期消费券发放时间为8月5日10：00-8月13日24：00，第二期消费券发放时间为8月14日-8月20日，第三期8月21日-8月27日，第四期8月28日-9月3日。实时定位在上海的消费者，可以在发放时间内进入美团、大众点评App，搜索“上海体育消费券”进行领取。为满足消费者更多个性化的需求，本轮体育消费券活动准备了满200减80、满120减50、满60减30、满40减20、满20减10和满10减5共六个面额的消费券，消费者可按需领用，先到先得。每位消费者每期最多可领取3张消费券，且每位消费者同时最多可持有3张。据“上海体育”公众号介绍，本次体育消费券适用场景多、覆盖范围广、优惠力度大。在发布会上，成都市体育局副局长陈志介绍，以成都大运会筹办举办为契机，成都积极开展“爱成都·迎大运”“运动成都·悦动生活”“万千商家齐参与”等主题体育消费促进活动，发放各类体育消费券和惠民运动券，促进体育消费持续稳步增长。2022年成都体育消费总规模为578.6亿元，居民人均体育消费为2720.6元。      ▲8月4日，成都大运会体操项目女子个人全能决赛看台上，观众为比赛队员加油 资料配图 摄影 陶轲  为持续激发体育消费活力和增长潜力，下一步，成都将持续深化体育消费试点工作，积极推进体育消费提质扩容。启动户外运动季活动，发布十大最受欢迎时尚运动消费场景。  具体而言，陈志介绍说，成都将加快推动“体育＋会展＋消费”平台建设，办好中国（成都）生活体育大会、“巴山蜀水·运动川渝”体育旅游休闲消费季、世界赛事名城发展大会、中国国际体育用品博览会等重大体育展会活动，为城市体育消费增长提供更多资源链接。

回答：上海市体育局联合美团、大众点评发放了总额500万元的体育消费券，覆盖3000多家本地运动门店，并设置了不同面额的消费券供消费者领取。而成都市体育局则是利用成都大运会的契机发放各类体育消费券和惠民运动券，同时计划通过举办大型体育展会活动和推动“体育＋会展＋消费”平台建设来进一步促进体育消费的提质扩容。

### Search results: {context}

### Question: {question}

请给出你的回答（回答的文本写在<response></response>之间。
"""
    if mode == "en":
        template = """### You are a helpful, respectful and honest assistant to help the user with questions. \
Please refer to the search results obtained from the local knowledge base. \
But be careful to not incorporate the information that you think is not relevant to the question. \
If you don't know the answer to a question, please don't share false information. \n
### Search results: {context} \n
### Question:"""

    if mode == "en_test":
        template = """### You are a helpful, respectful and honest assistant to help the user with questions. \
Please refer to the search results obtained from the local knowledge base. \
But be careful to not incorporate the information that you think is not relevant to the question. \
If you don't know the answer to a question, please don't share false information. \n
### Search results: {context} \n
### Question: {question} \n
### Answer:"""

    return template


def retriever_content(redis_conn):
    logging.info("CHECKING DB CONTENT------------------------------------------")    
    num_docs = int(redis_conn.ft("rag-redis").info()['num_docs'])
    page_size = 5000
    total_page = math.ceil(num_docs / page_size)
    logging.info(f"num_docs: {num_docs}, total_page: {total_page}, page_size:{page_size}")
    redis_conn.execute_command("FT.CONFIG", "SET", "MAXSEARCHRESULTS", str(num_docs))

    query = Query("*").paging(0, page_size)
    cursor = 0

    doc_dict= {}
    with tqdm(total=num_docs, desc="Processing documents") as pbar:
        while True:
            results = redis_conn.ft('rag-redis').search(query)

            for doc in results.docs:
                # print(doc.id, doc.content)
                doc_dict[doc.id] = doc.content
                pbar.update(1)

            if len(results.docs) < page_size:
                print(f"End of results. The last page retrieved {len(results.docs)} documents.")
                break

            cursor += 1
            query.paging(cursor * page_size, page_size)
        
    return doc_dict


def offloading(doc_dict, model="Qwen/Qwen2-7B-Instruct", template="en", mode=None):
    svc_ip, port = get_service_cluster_ip("llm-dependency-svc")
    llm_url = f"http://{svc_ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }

    doc_index = 0
    for doc_id, doc_content in tqdm(doc_dict.items()):
        search_result = doc_content
        logging.info(f"doc_index: {doc_index}, doc_id: {doc_id}, doc_content: {doc_content}")
        doc_index += 1
        template = get_template("en")
        prompt = template.format(context=search_result)
        logging.info(f"prompt: \n {prompt}")
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_completion_tokens": 128,
        }

        response = requests.post(llm_url, headers=headers, json=data)
        
        if response.status_code == 200:
            if mode == "test":
                break;
            continue
            #print("Response Body:", response.text)
        else:
            logging.info("ERROR")
            break
        
            
def rag_test(doc_dict, model="Qwen/Qwen2-7B-Instruct", mode=None):
    svc_ip, port = get_service_cluster_ip("llm-dependency-svc")
    llm_url = f"http://{svc_ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }

    user_prompt = "please summarize above."
    for doc_id, doc_content in tqdm(doc_dict.items()):
        search_result = doc_content
        logging.info(f"doc_id: {doc_id}, doc_content: {doc_content}")
        template = get_template("en_test")
        prompt = template.format(context=search_result, question=user_prompt)
        logging.info(f"{prompt}")
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_completion_tokens": 256,
        }

        response = requests.post(llm_url, headers=headers, json=data)
        if response.status_code == 200:
            if mode == "test":
                break;
            continue
            #print("Response Body:", response.text)
        else:
            logging.info("ERROR")
            break
        

def e2e_request():
    svc_ip, port = get_service_cluster_ip("chatqna-backend-server-svc")
    url = f"http://{svc_ip}:{port}/v1/chatqna"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "Qwen/Qwen2-7B-Instruct",
        "messages": "what is artificial intelligence",
        "max_tokens": 256,
    }

    response = requests.post(url, headers=headers, json=data)

    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
    
# e2e_request()

def one_iter_test(doc_dict):
    offloading(doc_dict, mode="test")
    print("RAG TEST ------------------------------------------------")
    rag_test(doc_dict, mode="test")
    


svc_ip, port = get_service_cluster_ip("vector-db")
redis_url = f"redis://{svc_ip}:{port}"
redis_conn = Redis.from_url(redis_url)

vector_index = SearchIndex.from_existing(name="rag-redis", redis_client=redis_conn)
results = vector_index.search(query="*")
doc_dict = retriever_content(redis_conn)

# one_iter_test()

logging.info(f"len(doc_dict) = {len(doc_dict)} results.total = {results.total}")
#one_iter_test(doc_dict)

offloading(doc_dict)
