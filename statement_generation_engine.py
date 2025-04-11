import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, OWL

class ElectricityMarketSemanticTranslator:
    """
    A class to translate electricity market data into semantic statements.
    """
    
    def __init__(self, db_path):
        """Initialize with database path."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Define namespaces
        self.WEM = Namespace("http://example.org/wem/")
        self.EVENT = Namespace("http://example.org/event/")
        self.FIBO = Namespace("http://www.omg.org/spec/FIBO/")
        self.EMF = Namespace("http://example.org/emf/")  # Event Model F
        
        # Initialize graph
        self.g = Graph()
        self.g.bind("wem", self.WEM)
        self.g.bind("event", self.EVENT)
        self.g.bind("fibo", self.FIBO)
        self.g.bind("emf", self.EMF)
        
        # Create base ontology
        self._create_base_ontology()
    
    def _create_base_ontology(self):
        """Create the base ontology for the semantic model."""
        # Define event types
        self.g.add((self.EVENT.MarketEvent, RDF.type, OWL.Class))
        self.g.add((self.EVENT.PriceEvent, RDF.type, OWL.Class))
        self.g.add((self.EVENT.PriceIncreaseEvent, RDFS.subClassOf, self.EVENT.PriceEvent))
        self.g.add((self.EVENT.PriceDecreaseEvent, RDFS.subClassOf, self.EVENT.PriceEvent))
        self.g.add((self.EVENT.QuantityEvent, RDF.type, OWL.Class))
        self.g.add((self.EVENT.QuantityIncreaseEvent, RDFS.subClassOf, self.EVENT.QuantityEvent))
        self.g.add((self.EVENT.QuantityDecreaseEvent, RDFS.subClassOf, self.EVENT.QuantityEvent))
        
        # Define market services
        self.g.add((self.WEM.MarketService, RDF.type, OWL.Class))
        services = ['energy', 'contingencyRaise', 'contingencyLower', 'regulationRaise', 'regulationLower', 'rocof']
        for service in services:
            self.g.add((self.WEM[service], RDF.type, self.WEM.MarketService))
            self.g.add((self.WEM[service], RDFS.label, Literal(service)))
    
    def detect_events(self, df, column, threshold=2.0, min_value_change=1.0):
        """Detect significant changes in a time series."""
        # Calculate changes
        changes = df[column].diff()
        
        # Skip if not enough data or no variance
        if len(changes) < 10 or changes.std() == 0:
            return pd.DataFrame()
        
        # Calculate Z-scores of changes
        mean_change = changes.mean()
        std_change = changes.std()
        z_scores = (changes - mean_change) / std_change
        
        # Detect events where absolute z-score exceeds threshold
        # and absolute change exceeds min_value_change
        events = df[(abs(z_scores) > threshold) & (abs(changes) > min_value_change)].copy()
        
        if events.empty:
            return events
            
        events['event_type'] = np.where(changes[events.index] > 0, 'increase', 'decrease')
        events['magnitude'] = abs(z_scores[events.index])
        events['change'] = changes[events.index]
        events['previous_value'] = df[column].shift(1)[events.index]
        
        return events
    
    def create_price_event_statements(self, events_df, market_service):
        """Convert detected price events to semantic statements."""
        for idx, event in events_df.iterrows():
            # Create event URI
            event_id = f"price_event_{market_service}_{idx.strftime('%Y%m%d%H%M%S')}"
            event_uri = self.EVENT[event_id]
            
            # Event type
            self.g.add((event_uri, RDF.type, self.EVENT.MarketEvent))
            self.g.add((event_uri, RDF.type, self.EVENT.PriceEvent))
            
            # Event time
            self.g.add((event_uri, self.EVENT.hasTime, Literal(idx, datatype=XSD.dateTime)))
            
            # Market service
            self.g.add((event_uri, self.EVENT.hasMarketService, self.WEM[market_service]))
            
            # Price values
            self.g.add((event_uri, self.EVENT.hasPrice, Literal(event['price'], datatype=XSD.float)))
            self.g.add((event_uri, self.EVENT.hasPreviousPrice, Literal(event['previous_value'], datatype=XSD.float)))
            self.g.add((event_uri, self.EVENT.hasPriceChange, Literal(event['change'], datatype=XSD.float)))
            
            # Event classification
            if event['event_type'] == 'increase':
                self.g.add((event_uri, RDF.type, self.EVENT.PriceIncreaseEvent))
                self.g.add((event_uri, self.EVENT.hasDirection, self.EVENT.Increasing))
            else:
                self.g.add((event_uri, RDF.type, self.EVENT.PriceDecreaseEvent))
                self.g.add((event_uri, self.EVENT.hasDirection, self.EVENT.Decreasing))
            
            # Magnitude
            self.g.add((event_uri, self.EVENT.hasMagnitude, Literal(event['magnitude'], datatype=XSD.float)))
    
    def create_quantity_event_statements(self, events_df, market_service, facility):
        """Convert detected quantity events to semantic statements."""
        for idx, event in events_df.iterrows():
            # Create event URI
            event_id = f"quantity_event_{market_service}_{facility}_{idx.strftime('%Y%m%d%H%M%S')}"
            event_uri = self.EVENT[event_id]
            
            # Event type
            self.g.add((event_uri, RDF.type, self.EVENT.MarketEvent))
            self.g.add((event_uri, RDF.type, self.EVENT.QuantityEvent))
            
            # Event time
            self.g.add((event_uri, self.EVENT.hasTime, Literal(idx, datatype=XSD.dateTime)))
            
            # Market service and facility
            self.g.add((event_uri, self.EVENT.hasMarketService, self.WEM[market_service]))
            self.g.add((event_uri, self.EVENT.hasFacility, self.WEM[facility]))
            
            # Quantity values
            self.g.add((event_uri, self.EVENT.hasQuantity, Literal(event['quantity'], datatype=XSD.float)))
            self.g.add((event_uri, self.EVENT.hasPreviousQuantity, Literal(event['previous_value'], datatype=XSD.float)))
            self.g.add((event_uri, self.EVENT.hasQuantityChange, Literal(event['change'], datatype=XSD.float)))
            
            # Event classification
            if event['event_type'] == 'increase':
                self.g.add((event_uri, RDF.type, self.EVENT.QuantityIncreaseEvent))
                self.g.add((event_uri, self.EVENT.hasDirection, self.EVENT.Increasing))
            else:
                self.g.add((event_uri, RDF.type, self.EVENT.QuantityDecreaseEvent))
                self.g.add((event_uri, self.EVENT.hasDirection, self.EVENT.Decreasing))
            
            # Magnitude
            self.g.add((event_uri, self.EVENT.hasMagnitude, Literal(event['magnitude'], datatype=XSD.float)))
    
    def process_market_service(self, service_type='energy', date_range=None, threshold=2.5, min_value_change=1.0):
        """Process data for a specific market service and generate semantic statements."""
        # Build date range filter if provided
        date_filter = ""
        if date_range:
            start_date, end_date = date_range
            date_filter = f" AND interval_time BETWEEN '{start_date}' AND '{end_date}'"
        
        # Load price data
        price_query = f"""
        SELECT interval_time, price 
        FROM dispatch_prices
        WHERE market_service = '{service_type}'{date_filter}
        ORDER BY interval_time
        """
        price_df = pd.read_sql_query(price_query, self.conn, parse_dates=['interval_time'], index_col='interval_time')
        
        if not price_df.empty:
            # Detect price events
            price_events = self.detect_events(price_df, 'price', threshold, min_value_change)
            
            # Create semantic statements for price events
            self.create_price_event_statements(price_events, service_type)
        
        # Process facility quantity data
        facilities_query = f"""
        SELECT DISTINCT facility_name
        FROM dispatch_quantities
        WHERE market_service = '{service_type}'{date_filter}
        """
        facilities = pd.read_sql_query(facilities_query, self.conn)['facility_name'].tolist()
        
        for facility in facilities:
            facility_query = f"""
            SELECT interval_time, quantity 
            FROM dispatch_quantities
            WHERE market_service = '{service_type}' AND facility_name = '{facility}'{date_filter}
            ORDER BY interval_time
            """
            facility_df = pd.read_sql_query(facility_query, self.conn, parse_dates=['interval_time'], index_col='interval_time')
            
            if not facility_df.empty:
                # Detect quantity events
                quantity_events = self.detect_events(facility_df, 'quantity', threshold, min_value_change)
                
                # Create semantic statements for quantity events
                self.create_quantity_event_statements(quantity_events, service_type, facility)
    
    def detect_correlations(self, window_minutes=15):
        """Detect temporal correlations between events."""
        # Get all price and quantity events
        price_events = []
        quantity_events = []
        
        for event_type in [self.EVENT.PriceEvent, self.EVENT.QuantityEvent]:
            for event in self.g.subjects(RDF.type, event_type):
                event_time = None
                for time_obj in self.g.objects(event, self.EVENT.hasTime):
                    event_time = pd.Timestamp(time_obj.toPython())
                    break
                
                if event_time:
                    if event_type == self.EVENT.PriceEvent:
                        price_events.append((event, event_time))
                    else:
                        quantity_events.append((event, event_time))
        
        # Sort events by time
        price_events.sort(key=lambda x: x[1])
        quantity_events.sort(key=lambda x: x[1])
        
        # Detect correlations
        for price_event, price_time in price_events:
            window_start = price_time - timedelta(minutes=window_minutes)
            window_end = price_time + timedelta(minutes=window_minutes)
            
            for quantity_event, quantity_time in quantity_events:
                if window_start <= quantity_time <= window_end:
                    # Create correlation
                    correlation_id = f"correlation_{price_time.strftime('%Y%m%d%H%M%S')}_{quantity_time.strftime('%Y%m%d%H%M%S')}"
                    correlation_uri = self.EVENT[correlation_id]
                    
                    self.g.add((correlation_uri, RDF.type, self.EMF.Correlation))
                    self.g.add((correlation_uri, self.EMF.correlates, price_event))
                    self.g.add((correlation_uri, self.EMF.correlates, quantity_event))
                    self.g.add((correlation_uri, self.EMF.timeWindow, Literal(window_minutes, datatype=XSD.integer)))
    
    def process_all_services(self, date_range=None, threshold=2.5, min_value_change=1.0):
        """Process all market services and generate semantic statements."""
        services = ['energy', 'contingencyRaise', 'contingencyLower', 'regulationRaise', 'regulationLower', 'rocof']
        
        for service in services:
            self.process_market_service(service, date_range, threshold, min_value_change)
        
        # Detect correlations between events
        self.detect_correlations()
    
    def generate_human_readable_statements(self):
        """Generate human-readable statements from the semantic graph."""
        statements = []
        
        # Process price events
        for event in self.g.subjects(RDF.type, self.EVENT.PriceEvent):
            # Get event properties
            event_time = None
            price = None
            previous_price = None
            price_change = None
            market_service = None
            direction = None
            
            for time_obj in self.g.objects(event, self.EVENT.hasTime):
                event_time = pd.Timestamp(time_obj.toPython())
            
            for price_obj in self.g.objects(event, self.EVENT.hasPrice):
                price = price_obj.toPython()
            
            for prev_price_obj in self.g.objects(event, self.EVENT.hasPreviousPrice):
                previous_price = prev_price_obj.toPython()
            
            for change_obj in self.g.objects(event, self.EVENT.hasPriceChange):
                price_change = change_obj.toPython()
            
            for service_obj in self.g.objects(event, self.EVENT.hasMarketService):
                service_str = service_obj.split('/')[-1]
                market_service = service_str
            
            is_increase = (event, RDF.type, self.EVENT.PriceIncreaseEvent) in self.g
            direction = "increased" if is_increase else "decreased"
            
            if all([event_time, price, previous_price, price_change, market_service]):
                statement = (
                    f"At {event_time}, the price of {market_service} {direction} "
                    f"from ${previous_price:.2f} to ${price:.2f} "
                    f"(a change of ${abs(price_change):.2f})."
                )
                statements.append(statement)
        
        # Process quantity events
        for event in self.g.subjects(RDF.type, self.EVENT.QuantityEvent):
            # Get event properties
            event_time = None
            quantity = None
            previous_quantity = None
            quantity_change = None
            market_service = None
            facility = None
            direction = None
            
            for time_obj in self.g.objects(event, self.EVENT.hasTime):
                event_time = pd.Timestamp(time_obj.toPython())
            
            for qty_obj in self.g.objects(event, self.EVENT.hasQuantity):
                quantity = qty_obj.toPython()
            
            for prev_qty_obj in self.g.objects(event, self.EVENT.hasPreviousQuantity):
                previous_quantity = prev_qty_obj.toPython()
            
            for change_obj in self.g.objects(event, self.EVENT.hasQuantityChange):
                quantity_change = change_obj.toPython()
            
            for service_obj in self.g.objects(event, self.EVENT.hasMarketService):
                service_str = service_obj.split('/')[-1]
                market_service = service_str
            
            for facility_obj in self.g.objects(event, self.EVENT.hasFacility):
                facility_str = facility_obj.split('/')[-1]
                facility = facility_str
            
            is_increase = (event, RDF.type, self.EVENT.QuantityIncreaseEvent) in self.g
            direction = "increased" if is_increase else "decreased"
            
            if all([event_time, quantity, previous_quantity, quantity_change, market_service, facility]):
                statement = (
                    f"At {event_time}, the dispatch quantity of {facility} for {market_service} {direction} "
                    f"from {previous_quantity:.2f} MW to {quantity:.2f} MW "
                    f"(a change of {abs(quantity_change):.2f} MW)."
                )
                statements.append(statement)
        
        # Sort statements by time
        statements.sort()
        
        return statements
    
    def save_graph(self, output_path):
        """Save the RDF graph to a file."""
        self.g.serialize(destination=output_path, format="turtle")
    
    def save_statements(self, output_path):
        """Save human-readable statements to a file."""
        statements = self.generate_human_readable_statements()
        with open(output_path, 'w') as f:
            for statement in statements:
                f.write(statement + "\n")
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


# Example usage
if __name__ == "__main__":
    # Initialize translator
    translator = ElectricityMarketSemanticTranslator("wem_data.db")
    
    # Process data for all services on a specific date
    date_range = ("2024-01-21", "2024-01-22")
    translator.process_all_services(date_range)
    
    # Generate and save statements
    statements = translator.generate_human_readable_statements()
    for statement in statements[:10]:  # Print first 10 statements
        print(statement)
    
    # Save outputs
    translator.save_graph("wem_semantic_graph.ttl")
    translator.save_statements("wem_semantic_statements.txt")
    
    # Close connection
    translator.close()
