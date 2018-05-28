import json

def tojson(a,et):

## mesh2json is a function to write the wanted mesh data into a json file
# with input et and nome. et is the given problem type, Linear or Quadratic
#  a is the required amount of element"""

 data = {}
 data['nodes'] = []
 data['elements'] = []
 data['meshsize'] = []
 data['material'] = []

 L = 1
 E = 1
 A = 1

 if et == 'linear':

    for i in range(a+1):
            node_value = i * L / a
            data['nodes'].append([node_value])

    for i in range(a):
            data['elements'].append([i, i + 1])

    data['etype'] = 'l2'
    data['gauss'] = 2
    data['boundary'] = [[0, a], [0.0, 0.0]]
    data['meshsize'] = L / a
    data['load'] = [[0, a],[0, 0]]
    data["material"]= [E, A]

    with open("h{}elem-l.json".format(a), 'w') as outfile:
            json.dump(data, outfile)


 elif et == 'quadratic':

    for i in range(2 * a + 1):
        node_value = i * L / (2 * a)
        data['nodes'].append([node_value])
    data['boundary'] = [[0, i], [0.0, 0.0]]
    data['load'] = [[0, i],[0, 0]]

    for i in range(a):
        data['elements'].append([2 * i, 2 * i + 1, 2 * i + 2])

    data['etype'] = 'l3'
    data['gauss'] = 2
    data['meshsize'] = L / a

    data["material"]= [E, A]

    with open("h{}elem-Q.json".format(a), 'w') as outfile:
        json.dump(data, outfile)
