import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Send, User, Bot } from 'lucide-react';
import clsx from 'clsx';

interface Message {
  id: number;
  text: string;
  isBot: boolean;
  timestamp: Date;
}

interface ChatSidebarProps {
  isOpen: boolean;
  updateMap: (coordinates: Array<{name: string, lat: number, lon: number}>) => void;
}

export const ChatSidebar: React.FC<ChatSidebarProps> = ({ isOpen, updateMap }) => {
  const [isResizing, setIsResizing] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(400);
  const [messages, setMessages] = useState<Message[]>([
    { 
      id: 1, 
      text: "Hello! How can I help you find locations today?", 
      isBot: true,
      timestamp: new Date() 
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const startResizing = useCallback((e: React.MouseEvent) => {
    setIsResizing(true);
    e.preventDefault();
  }, []);

  const stopResizing = useCallback(() => {
    setIsResizing(false);
  }, []);

  const resize = useCallback((e: MouseEvent) => {
    if (isResizing && sidebarRef.current) {
      const newWidth = e.clientX;
      if (newWidth >= 300 && newWidth <= window.innerWidth * 0.6) {
        setSidebarWidth(newWidth);
      }
    }
  }, [isResizing]);

  React.useEffect(() => {
    window.addEventListener('mousemove', resize);
    window.addEventListener('mouseup', stopResizing);
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [resize, stopResizing]);

  // Send message to Flask backend
  const sendMessage = async () => {
    console.log("sendMessage: Sending message", input);
    if (!input.trim()) return;
    
    const userMessage = { 
      id: Date.now(), 
      text: input, 
      isBot: false,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true); // Show loading state

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.text }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response from server");
      }

      const data = await response.json();
      
      console.log("sendMessage: Received response from server", data);
      console.log("sendMessage: Coordinates received:", data.coordinates);
      setIsLoading(false); // Hide loading state
      const botMessage = { 
        id: Date.now(),
        text: data.response, 
        isBot: true,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
      
      // Check if coordinates exist and have the correct format
      if (data.coordinates && Array.isArray(data.coordinates) && data.coordinates.length > 0) {
        console.log("sendMessage: Updating map with coordinates:", data.coordinates);
        updateMap(data.coordinates);
      } else {
        console.log("sendMessage: No valid coordinates received");
      }
    } catch (error) {
      console.error("sendMessage: Error sending message", error);
      setIsLoading(false);
      const errorMessage = {
        id: Date.now(),
        text: "Error occurred! Please check if the server is running.",
        isBot: true,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };
  
  const handleSend = () => {
    sendMessage();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      if (!e.shiftKey) {
        // Enter without Shift: send message
        e.preventDefault();
        handleSend();
      }
      // Shift+Enter: create new line is handled by default textarea behavior
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Format message text to render bold and new lines
  const formatMessageText = (text: string) => {
    if (!text) return '';
    
    // Preserve consecutive spaces by replacing them with &nbsp;
    let formattedText = text.replace(/ {2,}/g, (match) => {
      return ' ' + '&nbsp;'.repeat(match.length - 1);
    });
    
    // Replace /n with line breaks
    formattedText = formattedText.replace(/\/n/g, '<br/>');
    
    // Replace **text** with <strong>text</strong> for bold formatting
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    return formattedText;
  };

  return (
    <div
      ref={sidebarRef}
      style={{ 
        width: sidebarWidth,
        backdropFilter: 'blur(15px)',
        backgroundColor: 'rgba(255, 255, 255, 0.5)',
        boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
        border: '1px solid rgba(255, 255, 255, 0.18)'
      }}
      className={clsx(
        'absolute left-0 top-0 h-full shadow-lg z-10',
        'sidebar-transition',
        'flex flex-col',
        'border-r border-gray-200',
        'rounded-tr-2xl rounded-br-2xl',
        { 'select-none': isResizing },
        { '-translate-x-full': !isOpen }
      )}
    >
      <div className="flex-1 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">
        <>
          {messages.map((message) => (
              <div
                key={message.id}
                className={clsx(
                  'mb-4 max-w-[80%] message-enter',
                  message.isBot ? 'mr-auto' : 'ml-auto',
                  'w-full flex',
                  message.isBot ? 'justify-start' : 'justify-end'
                )}
              >
                {message.isBot ? (
                  <div className="flex items-start gap-2">
                    <div className="h-8 w-8 rounded-full bg-[#E3EFFF] flex items-center justify-center flex-shrink-0 border border-[#D4E4FF]">
                      <Bot size={16} className="text-[#0052CC]" />
                    </div>
                    
                    <div className="flex flex-col">
                      <div 
                        className="p-3 break-words text-[#333333] rounded-2xl rounded-tl-sm whitespace-pre-wrap" style={{ backdropFilter: 'blur(10px)', backgroundColor: 'rgba(227, 239, 255, 0.6)', border: '1px solid rgba(212, 228, 255, 0.7)' }}
                        dangerouslySetInnerHTML={{ __html: formatMessageText(message.text) }}
                      />
                      <span className="text-xs text-[#777777] mt-1 self-start">
                        {formatTime(message.timestamp)}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-2">
                    <div className="flex flex-col">
                      <div 
                        className="p-3 break-words text-white rounded-2xl rounded-tr-sm whitespace-pre-wrap" style={{ backdropFilter: 'blur(10px)', backgroundColor: 'rgba(0, 82, 204, 0.85)', border: '1px solid rgba(59, 125, 255, 0.7)' }}
                        dangerouslySetInnerHTML={{ __html: formatMessageText(message.text) }}
                      />
                      <span className="text-xs text-[#777777] mt-1 self-end">
                        {formatTime(message.timestamp)}
                      </span>
                    </div>
                    
                    <div className="h-8 w-8 rounded-full bg-[#FF5252] flex items-center justify-center flex-shrink-0 border border-[#FF7A7A]">
                      <User size={16} className="text-white" />
                    </div>
                  </div>
                )}
              </div>
            ))}
            
            {isLoading && (
              <div className="flex items-start gap-2 mb-4 mr-auto">
                <div className="h-8 w-8 rounded-full bg-[#E3EFFF] flex items-center justify-center border border-[#D4E4FF]">
                  <Bot size={16} className="text-[#0052CC]" />
                </div>
                <div className="p-3 rounded-2xl rounded-tl-sm" style={{ backdropFilter: 'blur(10px)', backgroundColor: 'rgba(227, 239, 255, 0.6)', border: '1px solid rgba(212, 228, 255, 0.7)' }}>
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
        </>
      </div>
      
      <div className="p-4" style={{ borderTop: '1px solid rgba(255, 255, 255, 0.3)' }}>
        <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="flex-1 resize-none rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-[#0052CC] text-[#333333]" style={{ backdropFilter: 'blur(10px)', backgroundColor: 'rgba(255, 255, 255, 0.5)', border: '1px solid rgba(255, 255, 255, 0.3)' }}
              rows={1}
            />
            <button
              onClick={handleSend}
              className="text-white p-3 rounded-lg transition-colors hover-scale" style={{ backdropFilter: 'blur(10px)', backgroundColor: 'rgba(0, 82, 204, 0.85)', border: '1px solid rgba(59, 125, 255, 0.5)', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      
      <div
        className="absolute right-0 top-0 w-1 h-full cursor-col-resize hover:bg-[#0052CC] transition-colors" style={{ background: 'rgba(200, 200, 200, 0.5)' }}
        onMouseDown={startResizing}
      />
    </div>
  );
};