import React, { useState } from 'react';
import { ChatSidebar } from './components/ChatSidebar';
import { Navbar } from './components/Navbar';
import { Map } from './components/Map';

function App() {
  const [isOpen, setIsOpen] = useState(true);
  const [coordinates, setCoordinates] = useState<Array<{name: string, lat: number, lon: number}>>([]);
  const [mapCenter, setMapCenter] = useState<[number, number]>([18.5204, 73.8567]); // Pune coordinates
  const [mapZoom, setMapZoom] = useState(13);
  
  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };
  
  // Function to update map with new coordinates
  const updateMap = (newCoordinates: Array<{name: string, lat: number, lon: number}>) => {
    console.log("updateMap: Updating map with new coordinates", newCoordinates);
    setCoordinates(newCoordinates);
    if (newCoordinates.length > 0) {
      const firstCoord = newCoordinates[0];
      setMapCenter([firstCoord.lat, firstCoord.lon]);
      setMapZoom(12); // Zoom in
    }
  };
  
  return (
    <div className="flex flex-col h-screen text-[#333333]" style={{ background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' }}>
      {/* Part 1: Navbar - fixed height at top */}
      <div className="h-16 flex-shrink-0">
        <Navbar 
          isOpen={isOpen} 
          onToggle={toggleSidebar} 
        />
      </div>
      
      {/* Part 2: Map and Sidebar - takes remaining height */}
      <div className="flex-1 relative">
        {/* Map positioned absolutely to fill the container */}
        <div className="absolute inset-0" style={{ zIndex: 1 }}>
          <Map 
            coordinates={coordinates}
            center={mapCenter}
            zoom={mapZoom}
          />
        </div>
        
        {/* Sidebar positioned above map with higher z-index */}
        <ChatSidebar 
          isOpen={isOpen}
          updateMap={updateMap}
        />
      </div>
    </div>
  );
}

export default App;