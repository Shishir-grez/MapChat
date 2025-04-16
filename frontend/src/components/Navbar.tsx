import React from 'react';
import { MapPin, MessageSquare, PanelLeftClose, Menu } from 'lucide-react';

interface NavbarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ isOpen, onToggle }) => {
  return (
    <nav className="h-16 z-30 px-4 w-full" style={{ 
        backdropFilter: 'blur(15px)', 
        background: 'linear-gradient(to right, rgba(227, 239, 255, 0.7), rgba(212, 228, 255, 0.7))',
        boxShadow: '0 4px 12px rgba(0, 82, 204, 0.1)', 
        borderBottom: '1px solid rgba(59, 125, 255, 0.2)' 
      }}>
      <div className="h-full flex items-center gap-4">
        <button
          onClick={onToggle}
          className="p-2 rounded-lg transition-colors hover-scale" style={{ background: 'rgba(255, 255, 255, 0.3)', backdropFilter: 'blur(5px)' }}
        >
          {isOpen ? <PanelLeftClose size={24} className="text-[#0052CC]" /> : <Menu size={24} className="text-[#0052CC]" />}
        </button>
        <div className="flex items-center gap-2">
          <MapPin size={24} className="text-[#0052CC]" />
          <span className="text-xl font-semibold" style={{ background: 'linear-gradient(to right, #0052CC, #3B7DFF)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>MapChat</span>
        </div>
        
        <div className="ml-auto flex items-center gap-4">
          <button className="p-2 rounded-lg transition-colors hover-scale" style={{ background: 'rgba(255, 255, 255, 0.3)', backdropFilter: 'blur(5px)' }}>
            <MessageSquare size={20} className="text-[#0052CC]" />
          </button>
        </div>
      </div>
    </nav>
  );
};