import argparse
import github
import os

def bioimageio_upload(resource_id: str, package_url: str): #, token:str):
    # Get bioimage-io/collection repodistory
    g = github.Github(auth=github.Auth.Token(os.environ["GITHUB_PAT"]))
    repo = g.get_repo("bioimage-io/collection")
    
    # Get the stage.yaml workflow (CI/GitHub action)
    workflow = repo.get_workflow("stage.yaml")


    # Dispatch the GitHub action for the runner
    ref = repo.get_branch("main")

    print(f"The worflow 'bioimage-io/collection:stage.yml' is going to be dispatched with resource_id={resource_id} and package_url={package_url}")
    
    ok = workflow.create_dispatch(
        ref=ref,
        inputs={
            "resource_id": resource_id,
            "package_url": package_url,
        },
    )

    # Assert that everything ran correctly
    assert ok

def main():
    parser = argparse.ArgumentParser(description="Programmatic staging of a new resource version (for advanced/internal use only).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--id", help="ID of the BioImage.IO collection that wou will stage (e.g. 'affable-shark').")
    parser.add_argument("-u", "--url", help="URL that points to the resource package ZIP that you want to stage.")
    # parser.add_argument("-t", "--token", help="GitHub token to authenticate.")
    args = vars(parser.parse_args())

    bioimageio_upload(resource_id=args['id'], package_url=args['url']) #, token=args['token'])

if __name__ == "__main__":
    main()