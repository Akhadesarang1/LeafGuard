import { motion } from "framer-motion";
import { ArrowLeft, Moon, RefreshCw, Sun } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";


// Error Boundary Component to catch errors gracefully
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Error caught by ErrorBoundary:", error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-red-100 p-4">
          <h2 className="text-red-600 text-2xl mb-4">
            Something went wrong. Please try again later.
          </h2>
          <button
            onClick={this.handleReload}
            className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
          >
            <RefreshCw className="w-5 h-5" />
            <span>Reload Page</span>
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function Blogs() {
  const navigate = useNavigate();
  const [darkMode, setDarkMode] = useState(() => {
    const storedPref = localStorage.getItem("darkMode");
    return storedPref ? JSON.parse(storedPref) : false;
  });

  // State to detect mobile devices
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    handleResize(); // Check on mount
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Simulate data loading
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    localStorage.setItem("darkMode", JSON.stringify(darkMode));
  }, [darkMode]);

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  const blogPosts = [
    {
      title: "Neem: Nature's Miracle",
      content:
        "Neem has powerful antifungal and antibacterial properties. It acts as a natural pesticide, protecting plants from harmful pests while promoting sustainable farming practices. Additionally, it’s valued for its healing benefits in traditional medicine.",
      benefits: [
        "Natural pesticide",
        "Boosts soil fertility",
        "Medicinal benefits",
      ],
      image: "/neem.jpg",
    },
    {
      title: "Guava: A Nutrient Powerhouse",
      content:
        "Guava is loaded with vitamin C, antioxidants, and fiber. It not only enhances plant health but also contributes to stronger immunity and improved digestion when included in your diet.",
      benefits: ["Rich in Vitamin C", "Improves digestion", "Boosts immunity"],
      image: "/guava.jpeg",
    },
    {
      title: "Tomato: Health Benefits and Care Tips",
      content:
        "Tomatoes are packed with lycopene, an antioxidant that supports heart health. They thrive in sunny, well-drained conditions, and with proper care, they yield nutrient-rich fruits ideal for any garden.",
      benefits: [
        "High in antioxidants",
        "Supports heart health",
        "Easy to grow at home",
      ],
      image: "/tomato.jpg",
    },
    {
      title: "Money Plant: The Lucky Charm",
      content:
        "Money plants are celebrated for their air-purifying properties and are believed to bring prosperity. They’re low maintenance, making them perfect for both homes and offices.",
      benefits: ["Air purifier", "Low maintenance", "Symbol of prosperity"],
      image: "/money_plant.webp",
    },
  ];

  // Animate each card from the y axis
  const cardVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: { opacity: 1, y: 0 },
  };

  // Animate inner content elements
  const contentVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  };

  // Variant for animating titles (page and blog post titles)
  const titleVariants = {
    hidden: { opacity: 0, y: -10, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.5 } },
  };

  return (
    <div
      className={`relative font-poppins min-h-screen py-12 px-4 sm:px-6 lg:px-8 transition-colors duration-500 ${
        darkMode
          ? "bg-gray-900 text-white"
          : "bg-gradient-to-br from-green-50 to-white text-gray-900"
      }`}
    >
      {/* Dark Mode Toggle Button */}
      <button
        onClick={() => setDarkMode(!darkMode)}
        aria-label="Toggle Dark Mode"
        className="absolute top-6 right-6 p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition"
      >
        {darkMode ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
      </button>

      <header className="mb-12 text-center">
        <button
          onClick={() => navigate(-1)}
          aria-label="Go Back"
          className="flex items-center justify-center text-green-700 hover:text-green-900 transition font-medium mb-4"
        >
          <ArrowLeft className="w-6 h-6 mr-2" /> Back
        </button>
        <motion.h1
          variants={titleVariants}
          initial="hidden"
          animate="visible"
          className="text-3xl sm:text-4xl md:text-5xl font-bold text-green-800 dark:text-green-200 tracking-wide mb-6"
        >
          Our Blog Insights
        </motion.h1>
      </header>

      {loading ? (
        <div className="max-w-4xl mx-auto space-y-16">
          {blogPosts.map((post, index) => (
            <div key={index}>
              <div className="animate-pulse flex flex-col md:flex-row items-center bg-gray-200 dark:bg-gray-700 rounded-xl shadow-lg p-6 sm:p-8">
                <div className="w-full md:w-1/2 h-48 bg-gray-300 dark:bg-gray-600 rounded-xl"></div>
                <div className="w-full md:w-1/2 md:px-6 mt-6 md:mt-0 space-y-4">
                  <div className="h-6 bg-gray-300 dark:bg-gray-600 rounded w-3/4"></div>
                  <div className="space-y-2">
                    <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-full"></div>
                    <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-5/6"></div>
                    <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-2/3"></div>
                  </div>
                  <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-1/2"></div>
                </div>
              </div>
              {index < blogPosts.length - 1 && (
                <hr className="my-8 border-t-2 border-green-200 dark:border-green-700" />
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="max-w-4xl mx-auto space-y-16">
          {blogPosts.map((post, index) => (
            <article key={index}>
              <motion.div
                variants={cardVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                transition={{
                  delay: isMobile && index === 0 ? 0.5 : 0, // delay the first card on mobile
                  duration: 0.6,
                }}
                className={`flex flex-col md:flex-row items-center ${
                  index % 2 === 0 ? "" : "md:flex-row-reverse"
                } bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sm:p-8 transform transition-all hover:shadow-2xl hover:scale-105`}
              >
                <motion.img
                  src={post.image}
                  alt={post.title}
                  loading="lazy"
                  className="w-full md:w-1/2 rounded-xl object-cover"
                  variants={contentVariants}
                />
                <div className="w-full md:w-1/2 md:px-6 mt-6 md:mt-0">
                  <motion.h2
                    variants={titleVariants}
                    className="text-2xl sm:text-3xl font-semibold text-green-800 dark:text-green-300 mb-4"
                  >
                    {post.title}
                  </motion.h2>
                  <motion.p
                    variants={contentVariants}
                    className="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed"
                  >
                    {post.content}
                  </motion.p>
                  <motion.h3
                    variants={contentVariants}
                    className="text-xl font-medium text-green-700 dark:text-green-400 mb-2"
                  >
                    Benefits:
                  </motion.h3>
                  <motion.ul
                    variants={contentVariants}
                    className="list-disc list-inside text-green-700 dark:text-green-300 space-y-1"
                  >
                    {post.benefits.map((benefit, i) => (
                      <motion.li
                        key={i}
                        variants={contentVariants}
                        className="leading-relaxed"
                      >
                        {benefit}
                      </motion.li>
                    ))}
                  </motion.ul>
                </div>
              </motion.div>
              {index < blogPosts.length - 1 && (
                <hr className="my-8 border-t-2 border-green-200 dark:border-green-700" />
              )}
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

export default function BlogsWithBoundary() {
  return (
    <ErrorBoundary>
      <Blogs />
    </ErrorBoundary>
  );
}
