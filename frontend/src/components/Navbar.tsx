import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Menu } from "lucide-react";
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext'; 

const Navbar: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { isAuthenticated, logout } = useAuth(); 
  const navigate = useNavigate();

  const handleSignOut = () => {
    logout();
    navigate('/auth'); 
  };

  return (
    <nav className="fixed top-0 left-0 right-0 bg-white bg-opacity-95 backdrop-blur-sm z-50 border-b">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center">
          <Link to="/" className="text-2xl font-bold text-primary">
            <span className="gradient-text">CashWise</span>
            <span className="ml-1">AI</span>
          </Link>
        </div>
        
        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-6">
          <a href="/#features" className="text-sm font-medium hover:text-primary transition">Features</a>
          <a href="/#how-it-works" className="text-sm font-medium hover:text-primary transition">How It Works</a>
          <a href="/#pricing" className="text-sm font-medium hover:text-primary transition">Pricing</a>
          <a href="/#about" className="text-sm font-medium hover:text-primary transition">About</a>
          
          {isAuthenticated ? (
            <>
              <Button onClick={handleSignOut}>Sign Out</Button>
            </>
          ) : (
            <>
              <Link to="/auth">
                <Button variant="outline" className="ml-2">Sign In</Button>
              </Link>
              <Link to="/auth">
                <Button>Try Free</Button>
              </Link>
            </>
          )}
        </div>
        
        {/* Mobile menu button */}
        <div className="md:hidden">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            aria-label="Toggle menu"
          >
            <Menu />
          </Button>
        </div>
      </div>
      
      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden px-4 py-2 pb-4 bg-white border-t">
          <div className="flex flex-col space-y-4">
            <a href="/#features" className="text-sm font-medium hover:text-primary transition py-2">Features</a>
            <a href="/#how-it-works" className="text-sm font-medium hover:text-primary transition py-2">How It Works</a>
            <a href="/#pricing" className="text-sm font-medium hover:text-primary transition py-2">Pricing</a>
            <a href="/#about" className="text-sm font-medium hover:text-primary transition py-2">About</a>
            <div className="flex flex-col space-y-2 pt-2">
              
              {isAuthenticated ? (
                <Button onClick={handleSignOut} className="w-full">Sign Out</Button>
              ) : (
                <>
                  <Link to="/auth">
                    <Button variant="outline" className="w-full">Sign In</Button>
                  </Link>
                  <Link to="/auth">
                    <Button className="w-full">Try Free</Button>
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
