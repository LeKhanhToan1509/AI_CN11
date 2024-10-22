from db.connectMinio import ConnectMinio

class Bucket:
    def __init__(self):
        self.client = ConnectMinio().connect()

    def check_name(self, name: str):
        if name == "":
            return False
        for item in name:
            if not item.isalnum() or item.isdigit() or item.isupper():
                return False
        
        return True
            
    
    def check_exists(self, bucket_name: str):
        self.client = ConnectMinio().connect()
        found = self.client.bucket_exists(bucket_name)
        if not found:
            return False
        return True


        

        