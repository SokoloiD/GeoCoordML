
PET проект по определению типа территории по GPS координатам и изображению генплана

код  в папке online_inference/kubernetes_manifests
Сделано:
1) (5 баллов) Развернут kubernetes  в облаке https://cloud.google.com/kubernetes-engine


2) (4 балла) Написан простой pod manifests для  приложения ( online-inference-pod.yaml)
Скриншон приложен 

2а) (2 балл) Прописан  requests/limits т.к придожение требовательно к оперативной памяти. (online-inference-pod-resources.yaml)


3) (3 балла) Модифицировано приложение ( точнее модифицированы команды , передаваемые в докер контейнер) так , чьто приложение ждет 30 секунд и завершает свою работу через 60 секунд после старта
добавлены  liveness и readiness пробы. Результат работы виден в выдаче команды  kubectl get pods. Под первые 30-35 секне получает статус ready. Через 90 секунд (примерно) статус пода с ready меняется на completed  
файл online-inference-pod-probes.yaml 


4) (3 балла) Создан replicaset, сделаны 3 реплики  приложения.

Ответьте на вопрос, что будет, если сменить докер образа в манифесте и одновременно с этим 
а) уменьшить число реплик (число реплик уменьшается, версии остаются старыми)
б) увеличить число реплик. (число реплик увеличивается, только новые реплики имеют обновленную версию)

файл  online-inference-replicaset.yaml

5) (3 балла) Описан деплоймент приложения в 2х вариантах
 
a) Есть момент времени, когда на кластере есть как все старые поды, так и все новые (опишите эту ситуацию) (online-inference-deployment-blue-green.yaml) 
 - replicas: 3
 - maxSurge: 3
 - maxUnavailable: 0
б) одновременно с поднятием новых версии, гасятся старые (online-inference-deployment-rolling-update.yaml)
 - replicas: 3
 - maxSurge: 0
 - maxUnavailable: 1

Самопроверка 1 (есть за это балл)
Итого: 5+4+2+3+3+3+1= 21 балл. (или 22 если с баллом за самопроверку)

фото
![start-k8s](https://user-images.githubusercontent.com/46603429/122819878-ab671200-d2e3-11eb-8bee-3372159ffa01.png)
![google-cloud](https://user-images.githubusercontent.com/46603429/122819981-c8034a00-d2e3-11eb-8e48-bb38b9f4dc53.png)
