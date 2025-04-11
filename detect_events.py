import pandas as pd
import numpy as np
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD

def detect_events(df, column, threshold=2):
    """Detect significant changes in a time series."""
    # Calculate Z-scores of changes
    changes = df[column].diff()
    mean_change = changes.mean()
    std_change = changes.std()
    z_scores = (changes - mean_change) / std_change
    
    # Detect events where absolute z-score exceeds threshold
    events = df[abs(z_scores) > threshold].copy()
    events['event_type'] = np.where(z_scores > threshold, 'increase', 'decrease')
    events['magnitude'] = abs(z_scores)
    
    return events
