from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import argparse
from oauth2client.service_account import ServiceAccountCredentials

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-sh", "--shorttitle", help="The title of the short")
    parser.add_argument("-st", "--shortsfolder", nargs='?', const="subtitled_videos", help="Folder where the done shorts are stored", default="subtitled_videos")
    parser.add_argument("-mf", "--metadatafolder", nargs='?', const="metadata", help="Metadata folder", default="metadata")
    parser.add_argument("-df", "--drivefolder", nargs='?', const="Website", help="Folder name in Google Drive", default="Website")

    args = parser.parse_args()

    # Path to your service account key file
    service_account_file = f'{args.metadatafolder}/service_account.json'

    # Define the scope for the authorization
    scope = ['https://www.googleapis.com/auth/drive']

    # Authorize with the service account
    credentials = ServiceAccountCredentials.from_json_keyfile_name(service_account_file, scope)
    gauth = GoogleAuth()
    gauth.credentials = credentials

    drive = GoogleDrive(gauth)
    
    fileList = drive.ListFile().GetList()
    
    for file in fileList:
        if file['title'] == args.drivefolder:
            fileID = file['id']
    
    vid = drive.CreateFile({'title': f'{args.shorttitle}', 'parents': [{'id': f'{fileID}'}], 'mimeType': 'video/mp4'})
    vid.SetContentFile(f'{args.shortsfolder}/subtitled_{args.shorttitle}.mp4')
    vid.Upload()
    print('Created file %s with mimeType %s' % (vid['title'], vid['mimeType']))

if __name__ == "__main__":
    main()