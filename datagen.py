#!/usr/bin/env python
# coding: utf-8

#Code is adapted to latest update of DSTC SGDS on May 07, 2020

import json

class my_dictionary(dict): 
    def __init__(self): 
        self = dict() 

    def add(self, key, value): 
        self[key] = value 


folders = ['dev', 'train', 'test'] 

print('dstc_services:')
service_set = set()
for fol_name in folders:
    schema_file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/schema.json'
    with open (schema_file, 'r') as sch:
        schema = json.load(sch)
        
    for it in schema:
        service_set.add(it['service_name'])
        
with open('dstc-data/dstc_services.txt', 'w') as f:
    for item in list(service_set):
        f.write(str(item).lower() + "\n")
        
with open('dstc-data/dstc_services_allcaps.txt', 'w') as f:
    for item in list(service_set):
        f.write(str(item) + "\n")
print('done \n')


print('dstc_da_system and dstc_ds_user:')        
da_system = set()
da_usr = set()       
for fol_name in folders:
    if fol_name == 'dev':
        total = 20
    elif fol_name == 'train':
        total = 127
    elif fol_name == 'test':
        total = 34
    
    print(fol_name)
    
    for count in range (1, total+1):
        if count < 10:
            counter = str('00') + str(count)
        elif count < 100:
            counter = str('0') + str(count)
        elif count < 1000:
            counter = str(count)
            
        file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/dialogues_' + counter + '.json'
        
        with open(file, 'rb') as f:
            data = json.load(f)
            for dialog in data:
                for turn in dialog['turns']:
                    if turn['speaker'] == 'SYSTEM':
                        for action in turn['frames'][0]['actions']:
                            if not (action['slot']) :
                                slot = 'none'
                            else:
                                slot = action['slot']
                            da_system.add(str(turn['frames'][0]['service'] + "~" + str(action['act']) + "~" + str(slot)))

                    elif turn['speaker'] == 'USER':
                        for fr in turn['frames']:
                            for action in turn['frames'][0]['actions']:
                                if not (action['slot']):
                                    slot = 'none'
                                else:
                                    slot = action['slot']
                                da_usr.add(str(turn['frames'][0]['service'] + "~" + str(action['act']) + "~" + str(slot)))

with open('dstc-data/dstc_da_system.txt', 'w') as f:
    for item in list(da_system):
        f.write(str(item).lower() + "\n")

with open('dstc-data/dstc_da_user.txt', 'w') as f:
    for item in list(da_usr):
        f.write(str(item).lower() + "\n")
        
with open('dstc-data/dstc_da_system_gg.txt', 'w') as f:
    for item in list(da_system):
        f.write(str(item) + "\n")

with open('dstc-data/dstc_da_user_gg.txt', 'w') as f:
    for item in list(da_usr):
        f.write(str(item) + "\n")
print('done \n')

  
print('dstc_da_goal:')
da_goal = set()
  
for fol_name in folders:            
    file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/schema.json'
        
    with open(file, 'rb') as f:
        data = json.load(f)
        for dialog in data:
            service = dialog['service_name']
            for intent in dialog['intents']:
                inte = intent['name']
                for sl in intent['result_slots']:
                    da_goal.add(str(service) + "~" + str(inte) + "~" + str(sl))

with open('dstc-data/dstc_da_goal.txt', 'w') as f:
    for item in list(da_goal):
        f.write(str(item).lower() + "\n")

print('done \n')


print('dstc_ontology:')
arr = set()
  
for fol_name in folders:
    if fol_name == 'dev':
        total = 20
    elif fol_name == 'train':
        total = 127
    elif fol_name == 'test':
        total = 34
    
    print(fol_name)
    
    for count in range (1, total+1):
        if count < 10:
            counter = str('00') + str(count)
        elif count < 100:
            counter = str('0') + str(count)
        elif count < 1000:
            counter = str(count)
            
        file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/dialogues_' + counter + '.json'
                
        with open(file, 'rb') as f:
            data = json.load(f)
            for dialog in data:
                for turn in dialog['turns']:
                    if turn['speaker'] == 'USER':
                        for fr in turn['frames']:
                            intent = fr['state']['active_intent']
                            slots = fr['state']['slot_values'].keys()
                            for slot in slots:
                                str_slot = str(slot)
                                for item in fr['state']['slot_values'][str_slot]:
                                    arr.add(str(fr['service']) + "~" + str(slot) + "~" + str(item))
                                
                    elif turn['speaker'] == 'SYSTEM':
                        for fr in turn['frames']:
                            serve = fr['service']
                            for action in fr['actions']:
                                slot = action['slot']
                                for item in action['values']:
                                    arr.add(str(serve) + "~" + str(slot) + "~" + str(item))

key1 = set()
for it in list(arr):
    service, slot, value = it.split('~')
    key1.add(service)

onto = dict([(key, []) for key in list(key1)])

on = my_dictionary()

for iteration in onto.keys():
    dicti = my_dictionary()
    for it in list(arr):
        service, slot, value = it.split('~')
        if service == iteration:
            dicti.add(slot, [])
    on.add(iteration, dicti)

for item in list(on.keys()):
    for subitem in on[item]:
        for it in list(arr):
            service, slot, value = it.split('~')
            if service == item and subitem == slot:
                on[item][subitem].append(value)

with open('dstc-data/dstc_ontology.json', 'w') as outfile:
      json.dump(on, outfile, indent=4)

print('done \n')

###DATABASE CREATION
print('dstc_db:')
arrdb = set()
for fol_name in folders:
    if fol_name == 'dev':
        total = 20
    elif fol_name == 'train':
        total = 127
    elif fol_name == 'test':
        total = 34  
        
    print(fol_name)
    
    for count in range (1, total+1):
        if count < 10:
            counter = str('00') + str(count)
        elif count < 100:
            counter = str('0') + str(count)
        elif count < 1000:
            counter = str(count)
            
        file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/dialogues_' + counter + '.json'
            
        with open(file, 'rb') as f:
            data = json.load(f)
            
            for dialog in data:
                for turn in dialog['turns']:                                
                    if turn['speaker'] == 'SYSTEM':
                        for fr in turn['frames']:
                            serve = fr['service']
                            for item in fr.keys():
                                if item == 'service_call':
                                    method = fr['service_call']['method']
                                    arrdb.add(str(serve) + "~" + str(method))
                                    
key2 = set()
for it in list(arrdb):
    service, method = it.split('~')
    key2.add(service)

servicesdict = dict([(key, []) for key in list(key2)])

dbdict = my_dictionary()

for iteration in servicesdict.keys():
    dicti = my_dictionary()
    for it in list(arrdb):
        service, method = it.split('~')
        if service == iteration:
            dicti.add(method, [])
    dbdict.add(iteration, dicti)                                             
                                        
for fol_name in folders:
    if fol_name == 'dev':
        total = 20
    elif fol_name == 'train':
        total = 127
    elif fol_name == 'test':
        total = 34
    
    print(fol_name)
    
    for count in range (1, total+1):
        if count < 10:
            counter = str('00') + str(count)
        elif count < 100:
            counter = str('0') + str(count)
        elif count < 1000:
            counter = str(count)
            
        file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/dialogues_' + counter + '.json'
        
        with open(file, 'rb') as f:
            data = json.load(f)                                        
            for dialog in data:
                for turn in dialog['turns']:                                
                    if turn['speaker'] == 'SYSTEM':
                        for fr in turn['frames']:
                            serve = fr['service']
                            for item in fr.keys():
                                if item == 'service_call':
                                    method = fr['service_call']['method']
                                    for db in fr['service_results']:
                                        dbdict[serve][method].append(db)

with open('dstc-data/dstc_db.json', 'w') as outfile:
      json.dump(dbdict, outfile, indent = 4)
print('done \n')


### GOAL GENERATOR
print('dstc_goals_*:')
with open ('dstc-data/dstc_services_allcaps.txt', 'r') as serv:
    content = serv.read()
    serv_list = content.split('\n')[:-1]

  
for fol_name in folders:
    if fol_name == 'dev':
        total = 20
    elif fol_name == 'train':
        total = 127
    elif fol_name == 'test':
        total = 34  
        
    print(fol_name)
    
    goal_file = 'dstc-data/dstc_goals_' + str(fol_name) + '.json'
    f_goals = open(goal_file, 'w+')
    f_goals.close()
    
    schema_file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/schema.json'
    with open (schema_file, 'r') as sch:
        schema = json.load(sch)
    
    l_goal_dict = []
    
    for count in range (1, total+1):
        if count < 10:
            counter = str('00') + str(count)
        elif count < 100:
            counter = str('0') + str(count)
        elif count < 1000:
            counter = str(count)
            
        file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/dialogues_' + counter + '.json'
            
        with open(file, 'rb') as f:
            data = json.load(f)
        
        for dialog in data:
            goal_dict = {}
            goal_dict['dialogue_id'] = dialog['dialogue_id']
            goal_dict['goals'] = {}
            for item in serv_list:
                goal_dict['goals'][item] = {}
                
            for services in dialog['services']:
                for it in schema:
                    if it['service_name'] == services:
                        for intent in it['intents']:
                            for item in intent['result_slots']:
                                goal_dict['goals'][services][item] = set()
                
            ##INSERTING THE VALUES ACCORDING TO CONVERSATION SET BY USER             
            for turn in dialog['turns']:
                if turn['speaker'] == 'USER':
                    for fr in turn['frames']:
                        intent = fr['state']['active_intent']
                        slots = fr['state']['slot_values'].keys()
                        for slot in slots:
                            str_slot = str(slot)
                            for item in fr['state']['slot_values'][str_slot]:
                                goal_dict['goals'][fr['service']][slot].add(item)
                                
                # elif turn['speaker'] == 'SYSTEM':
                #     for fr in turn['frames']:
                #         serve = fr['service']
                #         for action in fr['actions']:
                #             slot = action['slot']
                #             for item in action['values']:
                #                 goal_dict[serve][slot].append(item)
                                # arr.add(str(serve) + "~" + str(slot) + "~" + str(item))
            
            #conversion of set into list for saving file
            for services in dialog['services']:
                for it in schema:
                    if it['service_name'] == services:
                        for intent in it['intents']:
                            for item in intent['result_slots']:
                                goal_dict['goals'][services][item] = list(goal_dict['goals'][services][item])
                               
            l_goal_dict.append(goal_dict)
            
    with open(goal_file, 'a') as out:
        json.dump(l_goal_dict, out, indent = 4)
print('done \n')


#####COMBINING DATA
print('dstc_*:')
for fol_name in folders:
    if fol_name == 'dev':
        total = 20
    elif fol_name == 'train':
        total = 127
    elif fol_name == 'test':
        total = 34
    
    print(fol_name)
    
    dest_file = 'dstc-data/dstc_' + str(fol_name) + '.json'
    
    dest_f = open(dest_file, "w+")
    dest_f.close()
    
    dest_data = []
    
    for count in range (1, total+1):
        if count < 10:
            counter = str('00') + str(count)
        elif count < 100:
            counter = str('0') + str(count)
        elif count < 1000:
            counter = str(count)
            
        file = 'dstc8-schema-guided-dialogue-master/' + str(fol_name) + '/dialogues_' + counter + '.json'
            
        with open(file, 'rb') as f:
            data = json.load(f)
            
        for j in data:
            dest_data.append(j)
            
    with open(dest_file, 'a') as out:
        json.dump(dest_data, out, indent = 4)
        
        

fs = open('dstc-data/dstc_da_system_gg.txt', 'r')
data_sys = fs.read().split('\n')[:-1]

fu = open('dstc-data/dstc_da_user_gg.txt', 'r')
data_usr = fu.read().split('\n')[:-1]

slot_dict = {}
services = {'RentalCars_2', 'Movies_3', 'Services_2', 'Media_2', 'Restaurants_1', 'Homes_2', 'Payment_1', 'Calendar_1', 'Flights_1', 'Weather_1', 'Trains_1', 'Messaging_1', 'RentalCars_3', 'Music_2', 'Hotels_3', 'Buses_3', 'Music_3', 'Travel_1', 'Flights_3', 'Homes_1', 'Music_1', 'Alarm_1', 'Movies_2', 'Events_2', 'Media_1', 'RentalCars_1', 'Services_3', 'Buses_2', 'Flights_2', 'Hotels_4', 'Hotels_2', 'RideSharing_2', 'Events_3', 'Media_3', 'Services_4', 'Restaurants_2', 'Movies_1', 'Hotels_1', 'Buses_1', 'Banks_1', 'Events_1', 'Services_1', 'Flights_4', 'RideSharing_1', 'Banks_2'}

for dom in services:
    if dom not in slot_dict.keys():
        slot_dict[dom] = {}
        
for dom in services:
    if 'info' not in slot_dict[dom].keys():
        slot_dict[dom]['info'] = set()
    if 'reqt' not in slot_dict[dom].keys():
        slot_dict[dom]['reqt'] = set()
    if 'book' not in slot_dict[dom].keys():
        slot_dict[dom]['book'] = set()
            
for item in data_sys:
    dom, act, slot = item.split('~')
    if slot != 'none':
        if act == 'INFORM' or act == 'INFORM_COUNT' or act == 'OFFER':
            slot_dict[dom]['info'].add(slot)
        elif act == 'REQUEST' or act == 'REQ_MORE':
            slot_dict[dom]['reqt'].add(slot)
        elif act == 'NOTIFY_SUCCESS' or act == 'NOTIFY_FAILURE':
            slot_dict[dom]['book'].add(slot)
        
for item in data_usr:
    dom, act, slot = item.split('~')
    if slot != 'none':
        if act == 'INFORM' or act == 'INFORM_INTENT':
            slot_dict[dom]['info'].add(slot)
        elif act == 'REQUEST' or act == 'REQUEST_ALTS':
            slot_dict[dom]['reqt'].add(slot)
        
for dom in services:
    slot_dict[dom]['info'] = list(slot_dict[dom]['info'])
    slot_dict[dom]['reqt'] = list(slot_dict[dom]['reqt'])
    slot_dict[dom]['book'] = list(slot_dict[dom]['book'])
    
with open('dstc-data/dstc_slot_types.json', 'w') as outfile:
      json.dump([slot_dict], outfile, indent=4)
        
print('\n Completed data generation.')        
print('All the files can be found in folder: dstc-data')


 