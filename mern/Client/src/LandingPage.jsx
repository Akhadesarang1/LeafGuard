
import { motion } from "framer-motion";
import { useState } from "react";
import { Helmet } from "react-helmet";
import {
  FaBars,
  FaBrain,
  FaCheckCircle,
  FaHome,
  FaMobileAlt,
  FaRobot,
  FaShieldAlt,
  FaTimes,
  FaUserCheck,
  FaUserCircle,
} from "react-icons/fa";
import { Link, NavLink, useNavigate } from "react-router-dom";
import UserHistory from "./UserHistory";

// Define navigation items with "Analyze" as the last element and using FaRobot
const navItems = [
  { name: "Home", path: "/", icon: FaHome },
  { name: "Blogs", path: "/blogs", icon: FaBrain },
  { name: "About Us", path: "/about-us", icon: FaUserCheck },
  { name: "Analyze", path: "/analyze", icon: FaRobot },
];

// Blog posts array (using local images from the public folder)
const blogPosts = [
  {
    title: "Caring for Your Garden",
    description: "Discover essential tips to keep your garden thriving.",
    image: "/plant-image.jpg",
  },
  {
    title: "Identifying Plant Diseases",
    description: "Learn how to recognize and treat common plant ailments.",
    image: "/organic-farm.jpg",
  },
  {
    title: "Organic Farming Techniques",
    description: "Explore sustainable practices for organic farming.",
    image: "/garden.jpg",
  },
];

// Variants for staggered animations in blog cards
const blogContainer = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.2 } },
};

const blogItem = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

// FAQ data array with additional questions
const faqData = [
  {
    question: "How does LeafGuard detect plant diseases?",
    answer:
      "LeafGuard uses advanced AI algorithms trained on thousands of plant images to detect diseases quickly and accurately.",
  },
  {
    question: "Is my data secure?",
    answer:
      "Absolutely. We use robust security protocols to ensure your data remains private and secure.",
  },
  {
    question: "Can I use LeafGuard on my mobile device?",
    answer:
      "Yes, our platform is mobile friendly and optimized for a seamless experience on any device.",
  },
  {
    question: "What kind of plants can I analyze with LeafGuard?",
    answer:
      "LeafGuard can analyze a wide range of plants including vegetables, fruits, ornamental plants, and more.",
  },
  {
    question: "How fast is the analysis?",
    answer:
      "Our AI model processes your images in just a few seconds, providing quick and accurate insights.",
  },
  {
    question: "Do I need a special camera?",
    answer:
      "No, a standard smartphone camera works perfectly for capturing the necessary details for analysis.",
  },
];

// Variants for navbar animations
const navContainerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.2,
    },
  },
};

const navItemVariants = {
  hidden: { opacity: 0, x: -50 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.5 } },
};

// Profile icon and mobile toggle animation (delayed until after nav items)
const profileContainerVariants = {
  hidden: { opacity: 0, x: -50 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.5, delay: 0.8 },
  },
};

// Hero section animation starts after navbar sequence
const heroVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 1, delay: 1.5 },
  },
};

// Key Features array with additional prompt text for tooltip
const keyFeatures = [
  {
    title: "Easy to Use",
    description: "Upload an image and get instant results.",
    icon: FaRobot,
    prompt:
      "Our platform is designed with simplicity in mind. Just upload and get results instantly.",
  },
  {
    title: "Accurate AI Model",
    description: "Trained with thousands of plant images.",
    icon: FaBrain,
    prompt:
      "Our AI model is trained on a vast dataset ensuring reliable and accurate diagnostics.",
  },
  {
    title: "Expert Recommendations",
    description: "Get advice from agricultural experts.",
    icon: FaCheckCircle,
    prompt:
      "Receive personalized insights and recommendations from leading agricultural experts.",
  },
  {
    title: "Free & Secure",
    description: "Your data is safe, and the tool is free.",
    icon: FaShieldAlt,
    prompt:
      "Enjoy our free service with the assurance that your data is protected by top-notch security protocols.",
  },
  {
    title: "Mobile Friendly",
    description: "Use it on any device, anytime.",
    icon: FaMobileAlt,
    prompt: "Access LeafGuard on your smartphone or tablet, wherever you are.",
  },
  {
    title: "Trusted by Farmers",
    description: "Thousands rely on our AI for plant health.",
    icon: FaUserCheck,
    prompt:
      "Join a growing community of farmers and gardeners who trust our AI to monitor and improve plant health.",
  },
];

export default function LandingPage() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [faqOpen, setFaqOpen] = useState({});
  const [showUserHistory, setShowUserHistory] = useState(false);
  const navigate = useNavigate();

  const handleGetStarted = () => {
    const token = localStorage.getItem("token");
    if (token) {
      navigate("/analyze");
    } else {
      alert("Please login to use the Analyze page.");
      navigate("/login");
    }
  };

  const toggleFAQ = (index) => {
    setFaqOpen((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-emerald-200 text-gray-800 font-roboto relative">
      {/* SEO Meta Tags */}
<Helmet>
  {/* Basic & Essential Meta Tags */}
  <meta charSet="utf-8" />
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"
  />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <title>LeafGuard | AI Powered Plant Health</title>
  <meta
    name="description"
    content="LeafGuard uses AI to detect, diagnose, and prevent plant diseases. Get expert insights and protect your garden."
  />
  <meta
    name="keywords"
    content="AI plant disease detection, garden health, plant care, LeafGuard AI, plant disease prevention, smart gardening, AI for plants, crop health monitoring"
  />
  <meta name="robots" content="index, follow" />
  <meta name="author" content="LeafGuard Team" />
  <meta name="language" content="English" />
  <meta name="revisit-after" content="7 days" />
  <meta name="format-detection" content="telephone=no,email=no,address=no" />
  <meta name="theme-color" content="#ffffff" />

  {/* Advanced Mobile & App Meta Tags */}
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-status-bar-style" content="default" />
  <meta name="apple-mobile-web-app-title" content="LeafGuard" />
  <meta name="application-name" content="LeafGuard" />
  <meta name="msapplication-config" content="/browserconfig.xml" />
  <meta name="msapplication-TileColor" content="#ffffff" />

  {/* Open Graph Meta Tags for Social Sharing */}
  <meta property="og:title" content="LeafGuard | AI Powered Plant Health" />
  <meta
    property="og:description"
    content="AI-powered plant disease detection for healthier gardens."
  />
  <meta
    property="og:image"
    content="https://mern-test-client.onrender.com/LeafGaurd Logo.svg"
  />
  <meta property="og:type" content="website" />
  <meta
    property="og:url"
    content="https://mern-test-client.onrender.com"
  />
  <meta property="og:site_name" content="LeafGuard" />
  <meta property="og:locale" content="en_US" />
  <meta property="fb:app_id" content="YOUR_FB_APP_ID" />

  {/* Twitter Card Meta Tags */}
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="LeafGuard | AI Powered Plant Health" />
  <meta
    name="twitter:description"
    content="AI-powered plant disease detection for healthier gardens."
  />
  <meta
    name="twitter:image"
    content="https://mern-test-client.onrender.com/LeafGaurd Logo.svg"
  />
  <meta name="twitter:image:alt" content="LeafGuard logo banner" />
  <meta name="twitter:site" content="@LeafGuardAI" />
  <meta name="twitter:creator" content="@LeafGuardAI" />

  {/* Canonical & Alternate Links */}
  <link
    rel="canonical"
    href="https://mern-test-client.onrender.com"
  />
  <link
    rel="alternate"
    href="https://mern-test-client.onrender.com"
    hreflang="en-us"
  />

  {/* Favicon and Touch Icons */}
  <link
    rel="icon"
    href="https://mern-test-client.onrender.com/logo.jpg"
    type="image/jpeg"
  />
  <link
    rel="apple-touch-icon"
    href="https://mern-test-client.onrender.com/LeafGaurd Logo.svg"
  />
  <meta name="msapplication-TileImage" content="https://mern-test-client.onrender.com/LeafGaurd Logo.svg" />

  {/* Structured Data (JSON-LD) */}
  <script type="application/ld+json">
    {JSON.stringify({
      "@context": "https://schema.org",
      "@type": "WebSite",
      "name": "LeafGuard",
      "url": "https://mern-test-client.onrender.com",
      "description":
        "LeafGuard uses AI to detect, diagnose, and prevent plant diseases. Get expert insights and protect your garden.",
      "image": "https://mern-test-client.onrender.com/LeafGaurd Logo.svg",
      "publisher": {
        "@type": "Organization",
        "name": "LeafGuard Team",
        "logo": {
          "@type": "ImageObject",
          "url": "https://mern-test-client.onrender.com/LeafGaurd Logo.svg"
        }
      }
    })}
  </script>
</Helmet>
      {/* SEO Meta Tags */}
     

      {/* Navbar */}
      <nav
        className="bg-white shadow-md fixed w-full top-0 left-0 z-50"
        aria-label="Main Navigation"
      >
        <motion.div
          className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center relative"
          variants={navContainerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Logo and Title */}
          <motion.div variants={navItemVariants}>
            <Link to="/" className="flex items-center gap-2">
              <img
                src="/LeafGaurd Logo.svg"
                alt="LeafGuard Logo"
                className="w-10 h-10 object-cover"
              />
              <span className="text-2xl font-bold text-emerald-700">
                LeafGuard
              </span>
            </Link>
          </motion.div>
          {/* Navigation Links (Desktop) */}
          <motion.div
            variants={navItemVariants}
            className="hidden md:flex gap-6"
          >
            {navItems.map((item) => {
              const Icon = item.icon;
              const extraClasses =
                item.name === "Analyze"
                  ? "font-bold text-emerald-800 bg-emerald-200 px-3 py-1 rounded-lg"
                  : "";
              return (
                <motion.div key={item.name} variants={navItemVariants}>
                  <NavLink
                    to={item.path}
                    className={({ isActive }) =>
                      `flex items-center gap-1 transition font-medium ${extraClasses} ${
                        isActive
                          ? "text-emerald-600 border-b-2 border-emerald-600"
                          : "text-gray-700 hover:text-emerald-600"
                      }`
                    }
                    aria-label={item.name}
                  >
                    <Icon size={20} />
                    {item.name}
                  </NavLink>
                </motion.div>
              );
            })}
          </motion.div>
          {/* Profile Icon & Mobile Menu Toggle */}
          <motion.div
            variants={profileContainerVariants}
            className="flex items-center gap-4"
          >
            <button
              onClick={() => setShowUserHistory(!showUserHistory)}
              className="block"
              aria-label="User Profile"
            >
              <FaUserCircle size={28} className="text-emerald-700" />
            </button>
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden text-emerald-700"
              aria-label="Toggle Menu"
            >
              {isMenuOpen ? <FaTimes size={24} /> : <FaBars size={24} />}
            </button>
          </motion.div>
          {/* User History Dropdown */}
          {showUserHistory && (
            <div className="absolute top-16 right-6">
              <UserHistory onClose={() => setShowUserHistory(false)} />
            </div>
          )}
        </motion.div>
        {isMenuOpen && (
          <div className="md:hidden bg-white shadow-md">
            <div className="px-6 py-4 flex flex-col gap-3">
              {navItems.map((item) => {
                const Icon = item.icon;
                const extraClasses =
                  item.name === "Analyze"
                    ? "font-bold text-emerald-800 bg-emerald-200 px-3 py-1 rounded-lg"
                    : "";
                return (
                  <NavLink
                    key={item.name}
                    to={item.path}
                    onClick={() => setIsMenuOpen(false)}
                    className={({ isActive }) =>
                      `flex items-center gap-1 transition font-medium ${extraClasses} ${
                        isActive
                          ? "text-emerald-600 border-l-4 border-emerald-600 pl-2"
                          : "text-gray-700 hover:text-emerald-600"
                      }`
                    }
                    aria-label={item.name}
                  >
                    <Icon size={20} />
                    {item.name}
                  </NavLink>
                );
              })}
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section – Animation starts after navbar sequence */}
      <header
        className="relative flex flex-col items-center justify-center text-center p-6 md:p-10 mt-24 min-h-[80vh] bg-local md:bg-fixed bg-cover bg-center"
        style={{ backgroundImage: "url('/hero-image.jpg')" }}
      >
        {/* Black Overlay */}
        <div className="absolute inset-0 bg-black opacity-60"></div>
        <motion.div
          variants={heroVariants}
          initial="hidden"
          animate="visible"
          className="relative z-10 p-6 md:p-10"
        >
          <h1 className="text-3xl md:text-5xl font-bold text-white">
            LeafGuard: Empower Your Garden with AI
          </h1>
          <span className="text-lg md:text-2xl mt-4 text-white block">
            Unleash smart plant care and secure a thriving harvest.
          </span>
          <p className="text-base md:text-lg mt-4 text-white max-w-2xl mx-auto">
            Revolutionize your gardening experience with AI-driven insights.
            Nurture your plants and watch your garden flourish.
          </p>
          <div className="flex justify-center">
            <motion.button
              onClick={handleGetStarted}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="mt-6 bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-3 text-base md:text-lg rounded-xl shadow-lg transition"
              aria-label="Get Started"
            >
              Get Started
            </motion.button>
          </div>
        </motion.div>
      </header>

      {/* About Us Section with new image */}
      <section className="py-16 px-8 bg-emerald-100">
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-6xl mx-auto flex flex-col md:flex-row items-center gap-8"
        >
          <div className="md:w-1/2">
            <img
              src="https://images.unsplash.com/photo-1524486361537-8ad15938e1a3?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Z3JlZW5ob3VzZXxlbnwwfHwwfHx8MA%3D%3D"
              alt="Greenhouse view"
              className="rounded-xl shadow-lg w-full object-cover"
              loading="lazy"
            />
          </div>
          <div className="md:w-1/2 text-center md:text-left">
            <h2 className="text-3xl font-semibold text-emerald-700">
              About Us
            </h2>
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="mt-4 text-gray-700 italic"
            >
              At <strong>LeafGuard</strong>, we blend advanced AI technology
              with a passion for agriculture. Our dedicated team of researchers
              and developers strives to create innovative solutions that empower
              farmers, gardeners, and plant enthusiasts worldwide.
            </motion.p>
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="mt-4 text-gray-700 italic"
            >
              Our mission is to deliver real-time plant disease detection and
              actionable insights to ensure a healthier, more sustainable
              future. Join us on our journey toward smarter and more efficient
              agricultural practices.
            </motion.p>
          </div>
        </motion.div>
      </section>

      {/* Key Features Section with Tooltip on Hover */}
      <section className="py-16 px-8 bg-white">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-3xl font-semibold text-center text-emerald-700"
        >
          Key Features
        </motion.h2>
        <div className="grid md:grid-cols-3 gap-8 mt-10 max-w-6xl mx-auto">
          {keyFeatures.map((feature, index) => (
            <div key={index} className="relative group">
              <motion.div
                className="p-6 bg-emerald-50 rounded-xl shadow-md text-center hover:shadow-xl transition"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
              >
                <feature.icon className="text-emerald-600 text-5xl mx-auto" />
                <h3 className="text-xl font-semibold mt-4">{feature.title}</h3>
                <p className="text-gray-600 mt-2">{feature.description}</p>
              </motion.div>
              {/* Tooltip Prompt */}
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-1 bg-gray-800 text-white text-sm rounded opacity-0 transition-all duration-300 group-hover:opacity-100 group-hover:translate-y-0">
                {feature.prompt}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Blogs Section */}
      <section id="blogs" className="py-16 px-8 bg-white">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-3xl font-semibold text-center text-emerald-700"
        >
          Latest Blogs
        </motion.h2>
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-4 max-w-4xl mx-auto text-gray-700 text-center"
        >
          Explore our insightful articles on plant care, disease prevention, and
          organic farming.
        </motion.p>
        <motion.div
          className="mt-10 grid md:grid-cols-3 gap-8 max-w-6xl mx-auto"
          variants={blogContainer}
          initial="hidden"
          whileInView="visible"
        >
          {blogPosts.map((post, index) => (
            <motion.div
              key={index}
              className="bg-emerald-50 rounded-xl overflow-hidden shadow-md hover:shadow-xl transition"
              variants={blogItem}
            >
              <img
                src={post.image}
                alt={post.title}
                className="w-full h-48 object-cover"
                loading="lazy"
              />
              <div className="p-4 text-center">
                <h3 className="text-xl font-semibold text-emerald-700">
                  {post.title}
                </h3>
                <p className="text-gray-600 mt-2">{post.description}</p>
                <div className="flex justify-center">
                  <motion.button
                    onClick={() => navigate("/blogs")}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="mt-4 bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded transition"
                    aria-label="Read More"
                  >
                    Read More
                  </motion.button>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Analyze CTA Section */}
      <section className="py-16 px-8 bg-emerald-50 text-center">
        <motion.h2
          initial={{ opacity: 0, scale: 0.8 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="text-2xl font-semibold text-emerald-700"
        >
          Snap. Scan. Save Your Plants!!!
        </motion.h2>
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-4 text-gray-700 max-w-2xl mx-auto"
        >
          "Just upload a picture, and let our AI detect diseases before they
          spread. Your plants deserve the best care!"
          <br />
          Please login to access the Analyze page and begin your plant diagnosis
          journey.
        </motion.p>
        <div className="flex justify-center">
          <motion.button
            onClick={handleGetStarted}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-6 bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-3 text-lg rounded-xl shadow-lg flex items-center justify-center gap-2 transition"
            aria-label="Go to Analyze"
          >
            <FaRobot size={20} />
            Go to Analyze
          </motion.button>
        </div>
      </section>

      {/* Testimonial Section */}
      <section className="py-16 px-8 bg-white">
        <motion.h2
          className="text-3xl font-semibold text-center text-emerald-700"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          What Our Users Say
        </motion.h2>
        <motion.div
          className="mt-10 max-w-4xl mx-auto grid md:grid-cols-2 gap-8"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className="p-6 bg-emerald-50 rounded-xl shadow-md">
            <p className="text-gray-700 italic">
              "LeafGuard has revolutionized the way I care for my garden. The AI
              analysis is fast and accurate!"
            </p>
            <p className="mt-4 font-semibold text-emerald-700">- Jamie L.</p>
          </div>
          <div className="p-6 bg-emerald-50 rounded-xl shadow-md">
            <p className="text-gray-700 italic">
              "An indispensable tool for any gardener. It’s like having an
              expert with you at all times."
            </p>
            <p className="mt-4 font-semibold text-emerald-700">- Alex P.</p>
          </div>
          <div className="p-6 bg-emerald-50 rounded-xl shadow-md">
            <p className="text-gray-700 italic">
              "I couldn't believe the accuracy of the diagnosis. LeafGuard saved
              my garden!"
            </p>
            <p className="mt-4 font-semibold text-emerald-700">- Sarah W.</p>
          </div>
          <div className="p-6 bg-emerald-50 rounded-xl shadow-md">
            <p className="text-gray-700 italic">
              "Using LeafGuard has made plant care so much easier and fun. I
              highly recommend it."
            </p>
            <p className="mt-4 font-semibold text-emerald-700">- Michael T.</p>
          </div>
        </motion.div>
      </section>

      {/* FAQ Section */}
      <section className="py-16 px-8 bg-emerald-100">
        <motion.h2
          className="text-3xl font-semibold text-center text-emerald-700"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          Frequently Asked Questions
        </motion.h2>
        <div className="max-w-4xl mx-auto mt-10 space-y-4">
          {faqData.map((faq, index) => (
            <div key={index} className="border-b border-gray-300 pb-4">
              <button
                onClick={() => toggleFAQ(index)}
                className="w-full text-left flex justify-between items-center focus:outline-none"
                aria-expanded={faqOpen[index] ? "true" : "false"}
              >
                <span className="text-xl font-medium text-emerald-700">
                  {faq.question}
                </span>
                <span className="text-2xl text-emerald-700">
                  {faqOpen[index] ? "-" : "+"}
                </span>
              </button>
              {faqOpen[index] && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  transition={{ duration: 0.4 }}
                  className="mt-2 text-gray-700"
                >
                  {faq.answer}
                </motion.div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="text-center py-6 text-gray-700 bg-white shadow-md">
        <p>&copy; 2025 LeafGuard. All rights reserved.</p>
      </footer>
    </div>
  );
}
