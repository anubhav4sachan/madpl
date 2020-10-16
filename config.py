# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
@modified: anubhav sachan
"""

class Config():
    def __init__(self):
        self.domain = []
        self.intent = []
        self.slot = []
        self.da = []
        self.da_usr = []
        self.data_file = ''
        self.train_file = ''
        self.val_file = ''
        self.test_file = ''
        
        self.db_domains = []
        self.belief_domains = []
        
    def init_inform_request(self):
        self.inform_da = []
        self.request_da = []
        self.requestable = []
        
        for da in self.da_usr:
            d, i, s = da.split('-')
            if s == 'none':
                continue
            key = '-'.join([d,s])
            if i == 'inform' and key not in self.inform_da:
                self.inform_da.append(key)
            elif i == 'request' and key not in self.request_da:
                self.request_da.append(key)
                if d in self.db_domains and s != 'ref':
                    self.requestable.append(key)
                
        self.inform_da_usr = []
        self.request_da_usr = []
        for da in self.da_goal:
            d, i, s = da.split('-')
            key = '-'.join([d,s])
            if i == 'inform':
                self.inform_da_usr.append(key)
            else:
                self.request_da_usr.append(key)
        
    def init_dict(self):
        self.domain2idx = dict((a, i) for i, a in enumerate(self.belief_domains))
        self.idx2domain = dict((v, k) for k, v in self.domain2idx.items())

        self.inform2idx = dict((a, i) for i, a in enumerate(self.inform_da))
        self.idx2inform = dict((v, k) for k, v in self.inform2idx.items())
        
        self.request2idx = dict((a, i) for i, a in enumerate(self.request_da))
        self.idx2request = dict((v, k) for k, v in self.request2idx.items())
        
        self.inform2idx_u = dict((a, i) for i, a in enumerate(self.inform_da_usr))
        self.idx2inform_u = dict((v, k) for k, v in self.inform2idx_u.items())
        
        self.request2idx_u = dict((a, i) for i, a in enumerate(self.request_da_usr))
        self.idx2request_u = dict((v, k) for k, v in self.request2idx_u.items())
        
        self.requestable2idx = dict((a, i) for i, a in enumerate(self.requestable))
        self.idx2requestable = dict((v, k) for k, v in self.requestable2idx.items())
        
        self.da2idx = dict((a, i) for i, a in enumerate(self.da))
        self.idx2da = dict((v, k) for k, v in self.da2idx.items())
        
        self.da2idx_u = dict((a, i) for i, a in enumerate(self.da_usr))
        self.idx2da_u = dict((v, k) for k, v in self.da2idx_u.items())
        
    def init_dim(self):
        self.s_dim = len(self.da) + len(self.da_usr) + len(self.inform_da) + len(self.request_da) + len(self.belief_domains) + 6*len(self.db_domains) + 1#len(self.requestable) + 1
        self.s_dim_usr = len(self.da) + len(self.da_usr) + len(self.inform_da_usr)*2 + len(self.request_da_usr) + len(self.belief_domains)#*2
        self.a_dim = len(self.da)
        self.a_dim_usr = len(self.da_usr) + 1

class MultiWozConfig(Config):
    def __init__(self):
        self.domain = ['general', 'train', 'booking', 'hotel', 'restaurant', 'attraction', 'taxi', 'police', 'hospital']
        self.intent = ['inform', 'request', 'reqmore', 'bye', 'book', 'welcome', 'recommend', 'offerbook', 'nooffer', 'offerbooked', 'greet', 'select', 'nobook', 'thank']
        self.slot = ['none', 'name', 'area', 'choice', 'type', 'price', 'ref', 'leave', 'addr', 'phone', 'food', 'day', 'arrive', 'depart', 'dest', 'post', 'id', 'people', 'stars', 'ticket', 'time', 'fee', 'car', 'internet', 'parking', 'stay', 'department']
        self.da = ['general-reqmore-none-none', 'general-bye-none-none', 'booking-inform-none-none', 'booking-book-ref-1', 'general-welcome-none-none', 'restaurant-inform-name-1', 'hotel-inform-choice-1', 'train-inform-leave-1', 'hotel-inform-name-1', 'train-inform-id-1', 'restaurant-inform-choice-1', 'train-inform-arrive-1', 'restaurant-inform-food-1', 'train-offerbook-none-none', 'restaurant-inform-area-1', 'hotel-inform-type-1', 'attraction-inform-name-1', 'restaurant-inform-price-1', 'attraction-inform-area-1', 'train-offerbooked-ref-1', 'hotel-inform-area-1', 'hotel-inform-price-1', 'general-greet-none-none', 'attraction-inform-choice-1', 'train-inform-choice-1', 'hotel-request-area-?', 'attraction-inform-addr-1', 'train-request-leave-?', 'taxi-inform-car-1', 'attraction-inform-type-1', 'taxi-inform-phone-1', 'restaurant-inform-addr-1', 'attraction-inform-fee-1', 'restaurant-request-food-?', 'attraction-inform-phone-1', 'hotel-inform-stars-1', 'booking-request-day-?', 'train-inform-dest-1', 'train-request-depart-?', 'train-request-day-?', 'attraction-inform-post-1', 'hotel-recommend-name-1', 'restaurant-recommend-name-1', 'hotel-inform-internet-1', 'train-request-dest-?', 'attraction-recommend-name-1', 'restaurant-inform-phone-1', 'train-inform-depart-1', 'hotel-inform-parking-1', 'train-offerbooked-ticket-1', 'booking-book-name-1', 'hotel-request-price-?', 'train-inform-ticket-1', 'booking-nobook-none-none', 'restaurant-request-area-?', 'booking-request-people-?', 'hotel-inform-addr-1', 'train-request-arrive-?', 'train-inform-day-1', 'train-inform-time-1', 'booking-request-time-?', 'restaurant-inform-post-1', 'booking-book-day-1', 'booking-request-stay-?', 'restaurant-request-price-?', 'attraction-request-type-?', 'attraction-request-area-?', 'booking-book-people-1', 'restaurant-nooffer-none-none', 'taxi-request-leave-?', 'hotel-inform-phone-1', 'taxi-request-depart-?', 'restaurant-nooffer-food-1', 'hotel-inform-post-1', 'booking-book-time-1', 'train-request-people-?', 'attraction-inform-addr-2', 'taxi-request-dest-?', 'restaurant-inform-name-2', 'hotel-select-none-none', 'restaurant-select-none-none', 'booking-book-stay-1', 'train-offerbooked-id-1', 'hotel-inform-name-2', 'hotel-nooffer-type-1', 'train-offerbooked-people-1', 'taxi-request-arrive-?', 'attraction-recommend-addr-1', 'attraction-recommend-fee-1', 'hotel-recommend-area-1', 'hotel-request-stars-?', 'restaurant-nooffer-area-1', 'restaurant-recommend-food-1', 'restaurant-recommend-area-1', 'attraction-recommend-area-1', 'train-inform-leave-2', 'hotel-inform-choice-2', 'attraction-nooffer-area-1', 'attraction-nooffer-type-1', 'hotel-nooffer-none-none', 'hotel-recommend-price-1', 'attraction-inform-name-2', 'hotel-recommend-stars-1', 'restaurant-recommend-price-1', 'restaurant-inform-food-2', 'train-select-none-none', 'attraction-inform-type-2', 'booking-inform-name-1', 'hotel-inform-type-2', 'hotel-request-type-?', 'hotel-request-parking-?', 'hospital-inform-phone-1', 'hospital-inform-post-1', 'train-offerbooked-leave-1', 'attraction-select-none-none', 'hotel-select-type-1', 'taxi-inform-depart-1', 'hotel-inform-price-2', 'restaurant-recommend-addr-1', 'police-inform-phone-1', 'hospital-inform-addr-1', 'hotel-nooffer-area-1', 'hotel-inform-area-2', 'police-inform-post-1', 'police-inform-addr-1', 'attraction-recommend-type-1', 'attraction-inform-type-3', 'hotel-nooffer-stars-1', 'hotel-nooffer-price-1', 'taxi-inform-dest-1', 'hotel-request-internet-?', 'taxi-inform-leave-1', 'hotel-recommend-type-1', 'restaurant-inform-choice-2', 'hotel-recommend-internet-1', 'restaurant-select-food-1', 'restaurant-nooffer-price-1', 'train-offerbook-id-1', 'restaurant-inform-name-3', 'hotel-recommend-parking-1', 'attraction-inform-addr-3', 'attraction-recommend-post-1', 'attraction-inform-choice-2', 'restaurant-inform-area-2', 'train-offerbook-leave-1', 'hotel-inform-addr-2', 'restaurant-inform-price-2', 'attraction-recommend-phone-1', 'hotel-select-type-2', 'train-offerbooked-arrive-1', 'attraction-inform-area-2', 'hotel-recommend-addr-1', 'restaurant-select-food-2', 'train-offerbooked-depart-1', 'attraction-select-type-1', 'train-offerbook-arrive-1', 'taxi-inform-arrive-1', 'restaurant-inform-post-2', 'attraction-inform-fee-2', 'restaurant-inform-food-3', 'train-offerbooked-dest-1', 'attraction-inform-name-3', 'hotel-select-price-1', 'train-inform-arrive-2', 'attraction-request-name-?', 'attraction-nooffer-none-none', 'train-inform-ref-1', 'booking-book-none-none', 'police-inform-name-1', 'hotel-inform-stars-2', 'restaurant-select-price-1', 'attraction-inform-type-4']
        self.da_usr = ['general-thank-none', 'restaurant-inform-food', 'train-inform-dest', 'train-inform-day', 'train-inform-depart', 'restaurant-inform-price', 'restaurant-inform-area', 'hotel-inform-stay', 'restaurant-inform-time', 'hotel-inform-type', 'restaurant-inform-day', 'hotel-inform-day', 'attraction-inform-type', 'restaurant-inform-people', 'hotel-inform-people', 'hotel-inform-price', 'hotel-inform-stars', 'hotel-inform-area', 'train-inform-arrive', 'attraction-inform-area', 'train-inform-people', 'train-inform-leave', 'hotel-inform-parking', 'hotel-inform-internet', 'restaurant-inform-name', 'attraction-request-post', 'hotel-inform-name', 'attraction-request-phone', 'attraction-request-addr', 'restaurant-request-addr', 'restaurant-request-phone', 'attraction-inform-name', 'attraction-request-fee', 'general-bye-none', 'train-request-ticket', 'taxi-inform-leave', 'taxi-inform-none', 'train-request-ref', 'taxi-inform-depart', 'restaurant-inform-none', 'restaurant-request-post', 'taxi-inform-dest', 'train-request-time', 'hotel-inform-none', 'taxi-inform-arrive', 'train-inform-none', 'hotel-request-addr', 'restaurant-request-ref', 'hotel-request-post', 'hotel-request-phone', 'hotel-request-ref', 'train-request-id', 'taxi-request-car', 'attraction-request-area', 'train-request-arrive', 'train-request-leave', 'attraction-inform-none', 'attraction-request-type', 'hotel-request-price', 'hotel-request-internet', 'hospital-inform-none', 'hotel-request-parking', 'restaurant-request-price', 'hotel-request-area', 'restaurant-request-area', 'hospital-request-post', 'hotel-request-type', 'restaurant-request-food', 'hospital-request-phone', 'general-greet-none', 'police-inform-none', 'police-request-addr', 'hospital-request-addr', 'hospital-inform-department', 'police-request-post', 'police-inform-name', 'hotel-request-stars', 'police-request-phone', 'taxi-request-phone']
        self.da_goal = ['restaurant-inform-day', 'restaurant-inform-people', 'restaurant-inform-area', 'restaurant-inform-food', 'restaurant-inform-time', 'restaurant-inform-price', 'restaurant-inform-name', 'hotel-inform-day', 'hotel-inform-parking', 'hotel-inform-type', 'hotel-inform-stay', 'hotel-inform-people', 'hotel-inform-area', 'hotel-inform-stars', 'hotel-inform-price', 'hotel-inform-name', 'hotel-inform-internet', 'attraction-inform-area', 'attraction-inform-name', 'attraction-inform-type', 'train-inform-arrive', 'train-inform-day', 'train-inform-depart', 'train-inform-leave', 'train-inform-people', 'train-inform-dest', 'taxi-inform-arrive', 'taxi-inform-leave', 'taxi-inform-depart', 'taxi-inform-dest', 'hospital-inform-department', 'restaurant-request-addr', 'restaurant-request-post', 'restaurant-request-area', 'restaurant-request-food', 'restaurant-request-price', 'restaurant-request-phone', 'hotel-request-parking', 'hotel-request-type', 'hotel-request-addr', 'hotel-request-post', 'hotel-request-stars', 'hotel-request-area', 'hotel-request-price', 'hotel-request-internet', 'hotel-request-phone', 'attraction-request-type', 'attraction-request-fee', 'attraction-request-addr', 'attraction-request-post', 'attraction-request-area', 'attraction-request-phone', 'train-request-arrive', 'train-request-ticket', 'train-request-leave', 'train-request-id', 'train-request-time', 'taxi-request-car', 'taxi-request-phone', 'police-request-addr', 'police-request-post', 'police-request-phone', 'hospital-request-addr', 'hospital-request-post', 'hospital-request-phone']
        self.data_file = 'annotated_user_da_with_span_full_patchName.json'
        self.ontology_file = 'value_set.json'
        self.db_domains = ['train', 'hotel', 'restaurant', 'attraction']
        self.belief_domains = ['train', 'hotel', 'restaurant', 'attraction', 'taxi', 'police', 'hospital']
        self.val_file = 'valListFile.json'
        self.test_file = 'testListFile.json'
        
        self.h_dim = 100
        self.hs_dim = 100
        self.ha_dim = 50
        self.hv_dim = 50 # for value function
        
        # da to db
        self.mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange'},
                        'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
                        'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'type': 'type'},
                        'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination', 'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
                        'taxi': {'car': 'taxi_type', 'phone': 'taxi_phone'},
                        'hospital': {'department': 'department', 'phone': 'phone'},
                        'police': {'addr': 'address', 'name': 'name', 'post': 'postcode'}}
        # goal to da
        self.map_inverse = {'restaurant': {'address': 'addr', 'area': 'area', 'day': 'day', 'food': 'food', 'name': 'name', 'people': 'people', 'phone': 'phone', 'postcode': 'post', 'pricerange': 'price', 'time': 'time'},
                            'hotel': {'address': 'addr', 'area': 'area', 'day': 'day', 'internet': 'internet', 'name': 'name', 'parking': 'parking', 'people': 'people', 'phone': 'phone', 'postcode': 'post', 'pricerange': 'price', 'stars': 'stars', 'stay': 'stay', 'type': 'type'},
                            'attraction': {'address': 'addr', 'area': 'area', 'entrance fee': 'fee', 'name': 'name', 'phone': 'phone', 'postcode': 'post', 'type': 'type'},
                            'train': {'arriveBy': 'arrive', 'day': 'day', 'departure': 'depart', 'destination': 'dest', 'duration': 'time', 'leaveAt': 'leave', 'people': 'people', 'price': 'ticket', 'trainID': 'id'},
                            'taxi': {'arriveBy': 'arrive', 'car type': 'car', 'departure': 'depart', 'destination': 'dest', 'leaveAt': 'leave', 'phone': 'phone'},
                            'hospital': {'address': 'addr', 'department': 'department', 'phone': 'phone', 'postcode': 'post'},
                            'police': {'address': 'addr', 'phone': 'phone', 'postcode': 'post'}}

        self.init_inform_request() # call this first!
        self.init_dict()
        self.init_dim()


class ConfigDSTC():
    def __init__(self):
        self.domain = []
        self.intent = []
        self.slot = []
        self.da = []
        self.da_usr = []
        self.train_file = ''
        self.val_file = ''
        self.test_file = ''

        self.db_domains = []
        self.belief_domains = []
        
    def init_inform_request(self):
        self.inform_da = []
        self.request_da = []
        self.requestable = []
        
        for dac in self.da:
            d, i, s = dac.split('~')
            if s == 'none':
                continue
            key = '~'.join([d,s])
            if (i == 'INFORM' or i == 'INFORM_COUNT' or i == 'OFFER') and key not in self.inform_da:
                self.inform_da.append(key)
            elif (i == 'REQUEST' or i == 'REQ_MORE') and key not in self.request_da:
                self.request_da.append(key)
                # if d in self.db_domains and s != 'ref':
                #     self.requestable.append(key)
                
        self.inform_da_usr = []
        self.request_da_usr = []
        for da in self.da_usr:
            d, i, s = da.split('~')
            key = '~'.join([d,s])
            if (i == 'INFORM' or i == 'INFORM_INTENT'):
                self.inform_da_usr.append(key)
            elif i == 'REQUEST' or i == 'REQUEST_ALTS':
                self.request_da_usr.append(key)
        
    def init_dict(self):
        self.domain2idx = dict((a, i) for i, a in enumerate(self.belief_domains))
        self.idx2domain = dict((v, k) for k, v in self.domain2idx.items())

        self.inform2idx = dict((a, i) for i, a in enumerate(self.inform_da))
        self.idx2inform = dict((v, k) for k, v in self.inform2idx.items())
        
        self.request2idx = dict((a, i) for i, a in enumerate(self.request_da))
        self.idx2request = dict((v, k) for k, v in self.request2idx.items())
        
        self.inform2idx_u = dict((a, i) for i, a in enumerate(self.inform_da_usr))
        self.idx2inform_u = dict((v, k) for k, v in self.inform2idx_u.items())
        
        self.request2idx_u = dict((a, i) for i, a in enumerate(self.request_da_usr))
        self.idx2request_u = dict((v, k) for k, v in self.request2idx_u.items())
        
        # self.requestable2idx = dict((a, i) for i, a in enumerate(self.requestable))
        # self.idx2requestable = dict((v, k) for k, v in self.requestable2idx.items())
        
        self.da2idx = dict((a, i) for i, a in enumerate(self.da))
        self.idx2da = dict((v, k) for k, v in self.da2idx.items())
        
        self.da2idx_u = dict((a, i) for i, a in enumerate(self.da_usr))
        self.idx2da_u = dict((v, k) for k, v in self.da2idx_u.items())
        
    def init_dim(self):
        # self.s_dim = len(self.da) + len(self.da_usr) + len(self.inform_da) + len(self.request_da) + len(self.belief_domains) + 6*len(self.db_domains) + 1#len(self.requestable) + 1
        self.s_dim = len(self.da) + len(self.da_usr) + 1#len(self.requestable) + 1
        # self.s_dim_usr = len(self.da) + len(self.da_usr) + len(self.inform_da_usr)*2 + len(self.request_da_usr) + len(self.belief_domains)#*2
        self.s_dim_usr = len(self.da) + len(self.da_usr)
        self.a_dim = len(self.da)
        self.a_dim_usr = len(self.da_usr) + 1

class DSTCSGDSConfig(ConfigDSTC):
    def __init__(self, data_dir):
        
        self.domain = ['Calendar', 'Hotels', 'RentalCars', 'Services', 'Trains', 'Music', 'Homes', 'Flights', 'Travel', 'Media', 'RideSharing', 'Alarm', 'Banks', 'Movies', 'Weather', 'Messaging', 'Payment', 'Buses', 'Restaurants', 'Events']
        self.intent = ['AddAlarm', 'GetTrainTickets', 'FindBus', 'BuyBusTicket', 'BookAppointment', 'RentMovie', 'FindProvider', 'GetRide', 'PlayMovie', 'ReserveOnewayFlight', 'FindAttractions', 'BuyEventTickets', 'ReserveCar', 'ShareLocation', 'FindTrains', 'GetAvailableTime', 'GetCarsAvailable', 'RequestPayment', 'GetEventDates', 'LookupSong', 'FindRestaurants', 'ReserveRoundtripFlights', 'SearchHotel', 'SearchOnewayFlight', 'ReserveHotel', 'SearchHouse', 'FindHomeByArea', 'BookHouse', 'LookupMusic', 'FindEvents', 'FindApartment', 'TransferMoney', 'GetEvents', 'AddEvent', 'MakePayment', 'FindMovies', 'GetTimesForMovie', 'PlayMedia', 'PlaySong', 'GetAlarms', 'ReserveRestaurant', 'SearchRoundtripFlights', 'GetWeather', 'CheckBalance', 'BuyMovieTickets', 'ScheduleVisit']
        self.slot = ['place_name', 'restaurant_name', 'pets_allowed', 'smoking_allowed', 'number_of_rooms', 'car_name', 'has_live_music', 'number_of_adults', 'cast', 'property_name', 'city', 'title', 'available_start_time', 'recipient_account_type', 'venue', 'actors', 'date_of_journey', 'number_of_beds', 'pickup_date', 'shared_ride', 'intent', 'price_per_day', 'therapist_name', 'add_insurance', 'end_date', 'event_location', 'pickup_location', 'fare_type', 'free_entry', 'directed_by', 'origin_station_name', 'to_station', 'seating_class', 'airlines', 'destination_airport_name', 'has_wifi', 'track', 'event_time', 'total_price', 'destination_station_name', 'destination_city', 'director', 'leaving_time', 'movie_title', 'precipitation', 'leaving_date', 'origin_city', 'departure_time', 'car_type', 'new_alarm_name', 'year', 'subtitle_language', 'event_date', 'class', 'pickup_time', 'from', 'amount', 'number_of_seats', 'stay_length', 'device', 'appointment_time', 'city_of_event', 'number_stops', 'price', 'subtitles', 'dentist_name', 'transfers', 'location', 'flight_class', 'theater_name', 'is_unisex', 'additional_luggage', 'refundable', 'furnished', 'album', 'price_range', 'appointment_date', 'song_name', 'transfer_amount', 'playback_device', 'average_rating', 'departure_date', 'available_end_time', 'show_type', 'star_rating', 'wind', 'show_time', 'recipient_name', 'category', 'new_alarm_time', 'destination', 'dropoff_date', 'check_in_date', 'arrives_next_day', 'ride_fare', 'time', 'party_size', 'aggregate_rating', 'cuisine', 'ride_type', 'genre', 'receiver', 'num_passengers', 'passengers', 'is_redeye', 'inbound_departure_time', 'where_to', 'payment_method', 'show_date', 'in_unit_laundry', 'artist', 'pickup_city', 'private_visibility', 'account_balance', 'outbound_departure_time', 'rent', 'address_of_location', 'is_nonstop', 'transfer_time', 'outbound_arrival_time', 'from_station', 'starring', 'to_location', 'number_of_tickets', 'type', 'percent_rating', 'from_location', 'recipient_account_name', 'has_vegetarian_options', 'trip_protection', 'date', 'journey_start_time', 'approximate_ride_duration', 'inbound_arrival_time', 'number_of_days', 'price_per_night', 'travelers', 'event_name', 'address', 'has_laundry_service', 'serves_alcohol', 'doctor_name', 'total', 'origin', 'destination_airport', 'hotel_name', 'balance', 'venue_address', 'group_size', 'alarm_time', 'return_date', 'rating', 'street_address', 'number_of_riders', 'contact_name', 'area', 'humidity', 'fare', 'temperature', 'price_per_ticket', 'origin_airport', 'start_date', 'to', 'origin_airport_name', 'movie_name', 'to_city', 'check_out_date', 'from_city', 'good_for_kids', 'has_garage', 'stylist_name', 'visit_date', 'number_checked_bags', 'subcategory', 'wait_time', 'number_of_baths', 'attraction_name', 'has_seating_outdoors', 'phone_number', 'offers_cosmetic_services', 'event_type', 'account_type', 'alarm_name', 'pets_welcome']
        
        self.dp = 0.4
        
        f = open((str(data_dir) + '/' + 'dstc_services.txt'), 'r')
        content = f.read()
        self.domain = content.split('\n')[:-1]
        f.close()
        
        f = open(str(data_dir) + '/' + 'dstc_da_system.txt', 'r')
        content = f.read()
        self.da = content.split('\n')[:-1]
        f.close()
        
        f = open(str(data_dir) + '/' + 'dstc_da_user.txt', 'r')
        content = f.read()
        self.da_usr = content.split('\n')[:-1]
        f.close()

        f = open(str(data_dir) + '/' + 'dstc_da_goal.txt', 'r')
        content = f.read()
        self.da_goal = content.split('\n')[:-1]
        f.close()
        
        self.train_file = 'dstc_train.json'
        self.valid_file = 'dstc_dev.json'
        self.test_file = 'dstc_test.json'
        
        self.train_goals_file = 'dstc_goals_train.json'
        self.valid_goals_file = 'dstc_goals_dev.json'
        self.test_goals_file = 'dstc_goals_test.json'        

        self.ontology_file = 'dstc_ontology.json'
        
        self.db_domains = ['Hotels', 'Music', 'Buses', 'Trains']
        
        # self.belief_domains = ['Calendar', 'Hotels', 'RentalCars', 'Services', 'Trains', 'Music', 'Homes', 'Flights', 'Travel', 'Media', 'RideSharing', 'Alarm', 'Banks', 'Movies', 'Weather', 'Messaging', 'Payment', 'Buses', 'Restaurants', 'Events']
        f = open((str(data_dir) + '/' + 'dstc_services_allcaps.txt'), 'r')
        content = f.read()
        self.belief_domains = content.split('\n')[:-1]
        f.close()

        self.h_dim = 100
        self.hs_dim = 100
        self.ha_dim = 50
        self.hv_dim = 50 # for value function
        
        # da to db
        self.mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange'},
                        'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
                        'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'type': 'type'},
                        'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination', 'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
                        'taxi': {'car': 'taxi_type', 'phone': 'taxi_phone'},
                        'hospital': {'department': 'department', 'phone': 'phone'},
                        'police': {'addr': 'address', 'name': 'name', 'post': 'postcode'}}
        # goal to da
        self.map_inverse = {'restaurant': {'address': 'addr', 'area': 'area', 'day': 'day', 'food': 'food', 'name': 'name', 'people': 'people', 'phone': 'phone', 'postcode': 'post', 'pricerange': 'price', 'time': 'time'},
                            'hotel': {'address': 'addr', 'area': 'area', 'day': 'day', 'internet': 'internet', 'name': 'name', 'parking': 'parking', 'people': 'people', 'phone': 'phone', 'postcode': 'post', 'pricerange': 'price', 'stars': 'stars', 'stay': 'stay', 'type': 'type'},
                            'attraction': {'address': 'addr', 'area': 'area', 'entrance fee': 'fee', 'name': 'name', 'phone': 'phone', 'postcode': 'post', 'type': 'type'},
                            'train': {'arriveBy': 'arrive', 'day': 'day', 'departure': 'depart', 'destination': 'dest', 'duration': 'time', 'leaveAt': 'leave', 'people': 'people', 'price': 'ticket', 'trainID': 'id'},
                            'taxi': {'arriveBy': 'arrive', 'car type': 'car', 'departure': 'depart', 'destination': 'dest', 'leaveAt': 'leave', 'phone': 'phone'},
                            'hospital': {'address': 'addr', 'department': 'department', 'phone': 'phone', 'postcode': 'post'},
                            'police': {'address': 'addr', 'phone': 'phone', 'postcode': 'post'}}

        self.init_inform_request() # call this first!
        self.init_dict()
        self.init_dim()