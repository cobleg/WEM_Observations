def create_semantic_statements(events_df, market_service, facility=None):
    """Convert detected events to semantic statements."""
    g = Graph()
    
    # Define namespaces
    WEM = Namespace("http://example.org/wem/")
    EVENT = Namespace("http://example.org/event/")
    FIBO = Namespace("http://www.omg.org/spec/FIBO/")
    
    g.bind("wem", WEM)
    g.bind("event", EVENT)
    g.bind("fibo", FIBO)
    
    for idx, event in events_df.iterrows():
        # Create event URI
        event_id = f"event_{market_service}_{idx.strftime('%Y%m%d%H%M%S')}"
        event_uri = EVENT[event_id]
        
        # Event type
        g.add((event_uri, RDF.type, EVENT.MarketEvent))
        
        # Event time
        g.add((event_uri, EVENT.hasTime, Literal(idx, datatype=XSD.dateTime)))
        
        # Market service
        g.add((event_uri, EVENT.hasMarketService, WEM[market_service]))
        
        # Facility if provided
        if facility:
            g.add((event_uri, EVENT.hasFacility, WEM[facility]))
        
        # Event classification
        if event['event_type'] == 'increase':
            g.add((event_uri, RDF.type, EVENT.PriceIncreaseEvent))
            g.add((event_uri, EVENT.hasDirection, EVENT.Increasing))
        else:
            g.add((event_uri, RDF.type, EVENT.PriceDecreaseEvent))
            g.add((event_uri, EVENT.hasDirection, EVENT.Decreasing))
        
        # Magnitude
        g.add((event_uri, EVENT.hasMagnitude, Literal(event['magnitude'], datatype=XSD.float)))
        
    return g
