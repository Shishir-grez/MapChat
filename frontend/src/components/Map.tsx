import React, { useEffect } from 'react';
import { MapContainer, TileLayer, ZoomControl, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix Leaflet icon issue
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

// Map view updater component - updates map center and zoom when props change
const MapViewUpdater = ({ center, zoom }: { center: [number, number], zoom: number }) => {
  const map = useMap();
  
  useEffect(() => {
    console.log('Updating map view to:', center, zoom);
    map.setView(center, zoom, { animate: true });
  }, [map, center, zoom]);
  
  return null;
};

// Custom marker component with popup
const LocationMarker = ({ location }: { location: {name: string, lat: number, lon: number} }) => {
  const map = useMap();
  const position: [number, number] = [location.lat, location.lon];
  
  useEffect(() => {
    console.log('Adding marker at position:', position, 'for location:', location.name);
    
    // Create a marker with popup
    const marker = L.marker(position)
      .addTo(map)
      .bindPopup(`<b>${location.name}</b>`)
      .openPopup();
    
    // Add a pulse effect when marker is added
    const pulseIcon = L.divIcon({
      className: 'map-pin-pulse',
      html: `<div class="pulse"></div>`,
      iconSize: [40, 40],
      iconAnchor: [20, 20]
    });
    
    const pulse = L.marker(position, { icon: pulseIcon }).addTo(map);
    
    // Remove pulse after animation completes
    setTimeout(() => {
      map.removeLayer(pulse);
    }, 2000);
    
    return () => {
      map.removeLayer(marker);
    };
  }, [map, position, location.name]);
  
  return null; // Using imperative Leaflet API instead of React-Leaflet's declarative API for better control
};

export const Map: React.FC<{ coordinates?: Array<{name: string, lat: number, lon: number}>, center?: [number, number], zoom?: number }> = ({ 
  coordinates = [], 
  center = [18.5204, 73.8567], // Pune coordinates
  zoom = 13 
}) => {
  console.log('Map rendering with coordinates:', coordinates);
  
  // Force re-render when coordinates change
  const markersKey = coordinates.map(c => `${c.name}-${c.lat}-${c.lon}`).join('-');
  
  return (
    <MapContainer
      center={center}
      zoom={zoom}
      className="h-full w-full"
      zoomControl={false}
      key={`map-container-${coordinates.length}`} // Force re-render when number of coordinates changes
    >
      {/* This component updates the map view when center/zoom props change */}
      <MapViewUpdater center={center} zoom={zoom} />
      
      <ZoomControl position="topright" />
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {/* Map through locations from props with a key to force re-render */}
      <div key={markersKey}>
        {coordinates.map((loc, index) => (
          <LocationMarker 
            key={`marker-${index}-${loc.lat}-${loc.lon}`} 
            location={loc} 
          />
        ))}
      </div>
    </MapContainer>
  );
};