
import React from 'react';
import { Github, Linkedin, Twitter } from "lucide-react";

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-gray-50 border-t">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-2">
            <div className="flex items-center mb-4">
              <a href="/" className="text-2xl font-bold">
                <span className="gradient-text">CashWise</span>
                <span className="ml-1">AI</span>
              </a>
            </div>
            <p className="text-gray-600 max-w-md">
              AI-powered cash flow forecasting and management for businesses and individuals. Make better financial decisions with machine learning.
            </p>
            <div className="flex space-x-4 mt-6">
              <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-cashwise-blue transition">
                <Twitter className="h-5 w-5" />
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-cashwise-blue transition">
                <Linkedin className="h-5 w-5" />
              </a>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-cashwise-blue transition">
                <Github className="h-5 w-5" />
              </a>
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-bold uppercase tracking-wider text-gray-500 mb-4">Product</h4>
            <ul className="space-y-2">
              <li><a href="#features" className="text-gray-600 hover:text-cashwise-blue transition">Features</a></li>
              <li><a href="#pricing" className="text-gray-600 hover:text-cashwise-blue transition">Pricing</a></li>
              <li><a href="#" className="text-gray-600 hover:text-cashwise-blue transition">API Documentation</a></li>
              <li><a href="#" className="text-gray-600 hover:text-cashwise-blue transition">Changelog</a></li>
            </ul>
          </div>
          
          <div>
            <h4 className="text-sm font-bold uppercase tracking-wider text-gray-500 mb-4">Company</h4>
            <ul className="space-y-2">
              <li><a href="#about" className="text-gray-600 hover:text-cashwise-blue transition">About Us</a></li>
              <li><a href="#" className="text-gray-600 hover:text-cashwise-blue transition">Privacy Policy</a></li>
              <li><a href="#" className="text-gray-600 hover:text-cashwise-blue transition">Terms of Service</a></li>
              <li><a href="#" className="text-gray-600 hover:text-cashwise-blue transition">Contact</a></li>
            </ul>
          </div>
        </div>
        
        <div className="border-t mt-12 pt-8 flex flex-col sm:flex-row justify-between items-center">
          <p className="text-gray-500 text-sm">
            &copy; {currentYear} CashWise AI. All rights reserved.
          </p>
          <p className="text-gray-500 text-sm mt-2 sm:mt-0">
            Made with AI-powered forecasting technology
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
