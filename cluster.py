import os
import requests
import gcsfs
from time import time, sleep
from typing import List, Union, Optional


class ClusterStatus:
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class coCluster:
    """
    Python based interface to Bulk Embed endpoint
    to allow access to the end point using Python
    calls.

    Example useage:
        be = coBulkembed(api_key='~/api_key', model='small')
        be.submit_job('gs://cohere-assets/demos/static/squad_train_with_answers.jsonl',
                      text_field='question')
        print(be.embed_files)
    """

    URL = "https://api.cohere.ai/cluster-jobs/"

    # TODO use a requests session to avoid overhead of establishing a connection for every GET request
    # TODO add truncate option, maybe model option should be in the submit_job call?
    def __init__(self, api_key=None):
        """
        Inputs:
            api_key [Str]: either the API key or path to a file that
                           contains the api key
        """
        default_key = "gyDdihIHXjEaienjbpPEQfIUp9GYGZqm9XK6O91R"
        # assert model in self.MODELS, f"{model} is not supported"
        # self.model = model

        # load the api key
        # TODO add code to check the key
        if api_key is None:
            print("key is none")
            if os.path.isfile(os.path.expanduser(api_key)):
                with open(os.path.expanduser(api_key), "r") as fid:
                    lines = fid.readlines()
            self.token = lines[0].strip()
        else:
            self.token = default_key

        # setup google cloud storage link
        self.gcs = gcsfs.GCSFileSystem(project="valued-sight-253418")

        self.job_id = None
        self.input_files = []

    def gen_auth(self):
        """
        Generate authentication header
        """
        return {"Authorization": f"Bearer {self.token}"}

    def check_status(self, job_id=None):
        """
        Poll the end point to see if the bulk embdding job is done
        If it is complete, sets the embedding_files attribute with
           the output files containing the embeddings
        Returns:
            GET request response object
        """
        if job_id:
            self.job_id = job_id

        r = requests.get(
            self.URL + self.job_id,
            headers=self.gen_auth(),
        ).json()

        if r["status"].lower() == "complete" and len(self.input_files) == 0:
            self.input_files = r["output_urls"]

        return r

    def submit_job(
        self,
        embeddings_file: Union[str, List[str]],
        sim_threshold: float,
        min_cluster_size: int,
        wait=True,
        timeout=10,
    ):
        """
        Submit cluster job
        Inputs:
          embeddings_file [Str]: file on GCS with embeddings to cluster

          wait [bool] : if set to True, this call will block until either timeout is
                        reached or the embedding job is complete. if False, this call will
                        return immediately and users can check the job status by calling
                        the check_status method (default: True)
          timeout [int, float]: timeout in minutes before this call returns, only relevant
                                if wait is set to True.  If timeout is <= 0, then will
                                wait until the job finishess (default: 10 [minutes])
        """
        self.input_files = []
        self.file = embeddings_file
        if isinstance(embeddings_file, str):
            input_files = [embeddings_file]
        # construct the bulk embed request paylod
        payload = {
            "embeddings_url": input_files[0],
            "similarity_threshold": sim_threshold,
            "min_cluster_size": min_cluster_size,
        }
        print(payload)
        # generate request header
        self.header = self.gen_auth()

        # submit POST request
        self.job = requests.post(
            self.URL,
            json=payload,
            headers=self.header,
        )

        # job id
        self.job_id = self.job.json()["job_id"]
        print(f"Job ID: {self.job_id}")

        self.wait_for_results(wait=wait, timeout=timeout)

    def wait_for_results(self, wait, timeout):
        timeout = timeout * 60.0  # minutes -> seconds

        ERROR = False
        r_stat = self.check_status(self.job_id)

        polling_interval = 2
        st_time = None
        if wait:
            st = time()
            while True:
                if timeout > 0:
                    if time() - st >= timeout:
                        break
                    sleep(timeout / 10)
                else:
                    # if no timeout is set, then pool every 2 seconds
                    # TODO use the percent complete to estimate the time left
                    # and (conservatively) set the sleep based on the ETA
                    sleep(polling_interval)

                # check the job status
                r_stat = self.check_status()
                # we need a status for uploading files - seperate from processing
                # print(f'Percent complete: {r_stat["percent_complete"]}')
                if r_stat["status"].lower() == ClusterStatus.PROCESSING:
                    pass
                    # needs to be tested more
                    # if st_time is None:
                    #    if r_stat['percent_complete'] > 0:
                    #        st_time = [time(), r_stat['percent_complete']]
                    # else:
                    #    cur_time = [time(), r_stat['percent_complete']]
                    #    delta_percent = cur_time[1] - st_time[1]
                    #    if delta_percent > 0:
                    #        # divide by 5 as a fudge factor
                    #        polling_interval = 100*(cur_time[0] - st_time[0])/delta_percent/5.0
                    #    else:
                    #        polling_interval = 2 # default to 2 sec between polling
                elif r_stat["status"].lower() == ClusterStatus.FAILED:
                    ERROR = True
                    break
                elif r_stat["status"].lower() == ClusterStatus.COMPLETE:
                    break

        if ERROR:
            print("Error processing request:")
            print(r_stat)
        elif r_stat["status"].lower() == ClusterStatus.PROCESSING:
            print(
                'Timed out waiting for job to finsh - use "check_status" method manually'
            )
            # print(f"Percentage complete = {r_stat['percent_complete']}")
        else:
            print("Job complete!")

            self.input_files = r_stat["output_urls"]
            print("Embeddings stored in following files:")
            print("\n".join(self.input_files))
            return r_stat


class coClusterTest(coCluster):
    def submit_job(self, *args, **kwargs):
        return {
            "job_id": "string",
            "status": "string",
            "output_clusters_url": "string",
            "output_outliers_url": "string",
            "clusters": [],
            "threshold": "float",
            "min_cluster_size": "int",
        }


"""
(.venv) (base) evren_cohere_com@evren-1:~/applied_ml_team/cluster$ curl --request POST \
> --url https://api.cohere.ai/cluster-jobs \
> --header 'Authorization: Bearer 7VxbevQnGErUNujArBAztWqeya7J0bMja3vY6Jhq' \
> --header 'Content-Type: application/json' \
> --data '{"model": "small", "input_file_url": "gs://cohere-dev-central-2/evren/test_data/test.jsonl",
>          "text_field": "text"}'
{"job_id":"83713907-6fdc-4b6b-95ab-484c46127131"}(.venv) (base) evren_cohere_com@evren-1:~/applied_ml_team/cluster$ 
(.venv) (base) evren_cohere_com@evren-1:~/applied_ml_team/cluster$ curl --request GET \
> --url https://api.cohere.ai/embed-jobs/83713907-6fdc-4b6b-95ab-484c46127131 \
> --header 'Authorization: Bearer 7VxbevQnGErUNujArBAztWqeya7J0bMja3vY6Jhq'
{"job_id":"83713907-6fdc-4b6b-95ab-484c46127131","status":"failed","created_at":"2023-01-26T05:14:44.980501Z","input_url":"gs://cohere-dev-central-2/evren/test_data/test.jsonl","model":"small-20220926","truncate":"RIGHT","percent_complete":0}(.venv) (base) evren_cohere_com@evren-1:~/applied_ml_team/cluster$ curl --request GET --url https://api.cohere.ai/embed-jobs/83713907-6fdc-4b6b-95ab-484c46127131 --header 'Authorization: Bearer 7VxbevQnGErUNujArBAztWqeya7J0bMja3vY6Jhq'
{"job_id":"83713907-6fdc-4b6b-95ab-484c46127131","status":"failed","created_at":"2023-01-26T05:14:44.980501Z","input_url":"gs://cohere-dev-central-2/evren/test_data/test.jsonl","model":"small-20220926","truncate":"RIGHT","percent_complete":0}(.venv) (base) evren_cohere_com@evren-1:~/applied_ml_team/cluster$ vim requirements.txt 

"""
"""
curl --location 'https://api.cohere.ai/cluster-jobs/' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--header 'Authorization: Bearer gyDdihIHXjEaienjbpPEQfIUp9GYGZqm9XK6O91R' \
--data '{
  "embeddings_url": "gs://cohere-assets/demos/chatbot-intent-dataset.jsonl",
  "threshold": 0.95, 
  "min_cluster_size": 10
}'
"""
