
import glob
import os
import pprint
import json
from genson import SchemaBuilder

path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../outputs")
jsons_array = []
builder = SchemaBuilder("http://json-schema.org/draft-04/schema#")
builder.add_schema({"type": "object", "properties": {}})
for filepath in glob.glob(os.path.join(path, '*.json')):
    with open(filepath, 'r') as myfile:
        readed_json = json.loads(myfile.read())
        builder.add_object(readed_json)
        myfile.close()


pprint.pprint(builder.to_schema(), indent=4)

print(builder.to_schema())