// CureLinks.jsx
import { motion } from "framer-motion";
import React, { useEffect, useState } from "react";
import { FaArrowLeft, FaChevronRight, FaInfoCircle } from "react-icons/fa";
import { useNavigate, useParams } from "react-router-dom";
import Slider from "react-slick";


// Variants for the interactive container (image and info panel)
const imageContainerVariants = {
  rest: { width: "100%" },
  hover: { width: "60%" },
};

const infoContainerVariants = {
  rest: { width: 0, opacity: 0 },
  hover: { width: "40%", opacity: 1 },
};

// Custom slider arrows (black by default, turn green on hover)
const NextArrow = (props) => {
  const { className, style, onClick } = props;
  return (
    <motion.div
      className={`${className} text-black`}
      style={{ ...style, display: "block", right: "10px", zIndex: 2 }}
      onClick={onClick}
      aria-label="Next Slide"
      whileHover={{ scale: 1.1, color: "#38a169" }}
      transition={{ duration: 0.2 }}
    >
      <FaChevronRight size={20} />
    </motion.div>
  );
};

const PrevArrow = (props) => {
  const { className, style, onClick } = props;
  return (
    <motion.div
      className={`${className} text-black`}
      style={{ ...style, display: "block", left: "10px", zIndex: 2 }}
      onClick={onClick}
      aria-label="Previous Slide"
      whileHover={{ scale: 1.1, color: "#38a169" }}
      transition={{ duration: 0.2 }}
    >
      <FaChevronRight size={20} style={{ transform: "rotate(180deg)" }} />
    </motion.div>
  );
};

const sliderSettings = {
  dots: true,
  infinite: true,
  speed: 500,
  slidesToShow: 3,
  slidesToScroll: 1,
  arrows: true,
  nextArrow: <NextArrow />,
  prevArrow: <PrevArrow />,
  responsive: [
    {
      breakpoint: 768,
      settings: { slidesToShow: 1 },
    },
  ],
};

// Define the common cure cards for every disease.
const commonCureCards = [
  {
    title: "Neem Cake Fertilizer",
    link: "https://a.co/d/c1Xzm4Z",
    cardImage: "/Card images/Neem Cake Fertilizer.jpg",
    hoverText: "Rich in natural fungicides that help control canker.",
  },
  {
    title: "Potassium Sulfate",
    link: "https://a.co/d/96isIdX",
    cardImage: "/Card images/Potassium Sulfate.jpg",
    hoverText: "Strengthens plant cell walls and boosts immunity.",
  },
  {
    title: "Compost Tea",
    link: "https://a.co/d/cByf0nC",
    cardImage: "/Card images/Compost Tea.jpg",
    hoverText: "Promotes beneficial soil microbes that suppress pathogens.",
  },
  {
    title: "Bordeaux Mixture",
    link: "https://a.co/d/2zeIqrK",
    cardImage: "/Card images/Bordeaux Mixture.jpg",
    hoverText: "A traditional copper-based fungicide for fungal diseases.",
  },
];

// Disease cure data with expanded info strings and updated image paths.
const diseaseCureData = {
  Guava_Canker: {
    carouselImage: "/Guava_disease1.jpeg",
    info:
      "Guava canker is a severe fungal infection affecting guava trees. It produces deep, sunken, and discolored lesions on both the fruit and bark. The disease spreads rapidly in humid environments, leading to significant yield losses and reduced fruit quality. Proper pruning, sanitation, and timely fungicide treatments are essential for effective management.",
    cards: commonCureCards,
  },
  Guava_Dot: {
    carouselImage: "/Guava_disease4.jpeg",
    info:
      "Guava Dot is characterized by small, discolored spots on the leaves of guava trees. Although less severe than canker, persistent infection can impair photosynthesis and reduce overall plant vigor over time.",
    cards: commonCureCards,
  },
  Guava_Mummification: {
    carouselImage: "/Guava_disease3.jpeg",
    info:
      "Guava Mummification causes the fruits to dry out and shrivel, leading to mummified remnants on the tree. This condition results in significant yield losses and poor fruit quality, requiring timely cultural practices and fungicide applications.",
    cards: commonCureCards,
  },
  Guava_Rust: {
    carouselImage: "/Guava_disease2.jpeg",
    info:
      "Guava Rust is a fungal disease that results in rust-colored pustules on the leaves and fruits. This infection causes premature leaf drop and weakens the tree over time. Adequate air circulation, proper pruning, and fungicide treatments can help manage the disease.",
    cards: commonCureCards,
  },
  Money_plant_Bacterial_wilt_disease: {
    carouselImage: "/Money_plant disease1.jpeg",
    info:
      "Money Plant Bacterial Wilt is caused by soil-borne bacteria, leading to rapid wilting, yellowing, and eventual plant death. The disease spreads quickly in warm, humid conditions. Implementing proper watering practices, sanitation, and bactericide treatments are critical to control its spread.",
    cards: commonCureCards,
  },
  "Money_plant_Manganese Toxicity": {
    carouselImage: "/Money_plant Disease2.jpeg", // Updated image path
    info:
      "Money Plant Manganese Toxicity occurs when excessive manganese accumulates in the soil, interfering with the uptake of other essential nutrients. Symptoms include interveinal chlorosis, leaf discoloration, and stunted growth. Regular soil testing and balanced fertilization can help mitigate this issue and improve plant health.",
    cards: commonCureCards,
  },
  Neem_Alternaria: {
    carouselImage: "/neem_disease1.jpeg",
    info:
      "Neem Alternaria is a fungal disease that produces dark, irregular spots on neem leaves. It reduces the plant's photosynthetic efficiency and overall strength, often leading to premature leaf drop. Timely fungicide applications and proper cultural practices are necessary for control.",
    cards: commonCureCards,
  },
  Neem_Dieback: {
    carouselImage: "/neem_disease2.jpeg",
    info:
      "Neem Dieback is characterized by the gradual wilting and browning of neem branches. The disease may result from environmental stress or secondary infections. Pruning affected branches and improving tree nutrition can help restore health.",
    cards: commonCureCards,
  },
  Neem_Leaf_Miners: {
    carouselImage: "/neem_disease3.jpeg",
    info:
      "Neem Leaf Miners are insect pests that burrow into the leaves, creating tunnels and causing significant damage. This infestation reduces the leaf's ability to photosynthesize and weakens the overall plant. Integrated pest management strategies are recommended.",
    cards: commonCureCards,
  },
  Neem_Leaf_Miners_Powdery_Mildew: {
    carouselImage: "/Neem_Leaf_Miners_Powdery_Mildew.jpeg", // Update if needed
    info:
      "This condition involves damage from leaf miners combined with a powdery mildew infection. The dual impact severely compromises leaf function and overall tree health. A combination of insecticidal and fungicidal treatments, along with improved cultural practices, is advised.",
    cards: commonCureCards,
  },
  Neem_Powdery_Mildew: {
    carouselImage: "/Gemini_Generated_Image_92wr5d92wr5d92wr.jpeg",
    info:
      "Neem Powdery Mildew is a fungal infection that appears as a white, powdery coating on neem leaves. This condition hampers photosynthesis and can lead to reduced growth and vigor. Effective fungicide treatments and enhanced air circulation are crucial for management.",
    cards: commonCureCards,
  },
  Tomato___Bacterial_spot: {
    carouselImage: "/Gemini_Generated_Image_tqqdl2tqqdl2tqqd.jpeg",
    info:
      "Tomato Bacterial Spot is caused by bacterial pathogens that produce dark, water-soaked lesions on tomato leaves and fruits. The disease can lead to significant yield losses if not managed properly. Copper-based sprays and resistant cultivars are commonly used to control the outbreak.",
    cards: commonCureCards,
  },
  Tomato___Early_blight: {
    carouselImage: "/Gemini_Generated_Image_tomc5utomc5utomc.jpeg",
    info:
      "Tomato Early Blight is a fungal disease that causes brown lesions on the lower leaves and stems of tomato plants. The infection gradually moves upward, resulting in defoliation and reduced fruit quality. Regular fungicide applications and crop rotation are recommended for control.",
    cards: commonCureCards,
  },
  Tomato___Late_blight: {
    carouselImage: "/Gemini_Generated_Image_tqqdl2tqqdl2tqqd.jpeg",
    info:
      "Tomato Late Blight is a severe fungal disease that causes rapid wilting, decay, and destruction of tomato plants. It affects both leaves and fruits and thrives in cool, damp conditions. Early detection and prompt fungicide treatments are essential to limit crop losses.",
    cards: commonCureCards,
  },
  Tomato___Leaf_Mold: {
    carouselImage: "/Tomato_Leaf_Mold.jpeg",
    info:
      "Tomato Leaf Mold is a fungal disease that appears as a velvety gray growth on the underside of tomato leaves. It reduces the plant's ability to photosynthesize and can lead to significant yield reductions if not treated.",
    cards: commonCureCards,
  },
};

const CureLinks = () => {
  const navigate = useNavigate();
  const { disease } = useParams(); // Retrieved disease from URL parameter
  const [loading, setLoading] = useState(true);
  const [isInfoVisible, setInfoVisible] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    setIsMobile(window.innerWidth < 768);
    const timer = setTimeout(() => setLoading(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  const renderSkeleton = () => (
    <div className="animate-pulse flex flex-col items-center justify-center min-h-screen bg-gradient-to-r from-green-100 to-blue-100 p-6">
      <div className="w-full max-w-2xl h-60 bg-gray-700 rounded-2xl mb-6" />
      <div className="w-3/4 h-6 bg-gray-700 rounded mb-4" />
      <div className="w-1/2 h-6 bg-gray-700 rounded mb-8" />
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 w-full max-w-2xl">
        {Array(4)
          .fill(0)
          .map((_, idx) => (
            <div key={idx} className="bg-gray-700 h-32 rounded-xl" />
          ))}
      </div>
    </div>
  );

  if (loading) return renderSkeleton();

  // Retrieve cure data for the detected disease.
  const cureData = diseaseCureData[disease];

  if (!cureData) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-r from-green-100 to-blue-100">
        <p className="text-xl text-red-600">
          No cure information available for this disease.
        </p>
        <button
          onClick={() => navigate("/")}
          className="mt-4 px-4 py-2 bg-green-500 text-white rounded-full"
          aria-label="Go Back"
        >
          Go Back
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-r from-green-100 to-blue-100 transition-all duration-500 py-6 px-4">
      <div className="max-w-screen-lg mx-auto">
        <motion.button
          onClick={() => navigate("/")}
          className="flex items-center mb-8 px-6 py-3 rounded-full bg-gradient-to-r from-green-500 to-green-700 text-white shadow-lg hover:shadow-2xl transition-all"
          aria-label="Back to Home"
          whileHover={{ scale: 1.05 }}
        >
          <FaArrowLeft size={20} className="mr-2" /> Back to Home
        </motion.button>
        <motion.h1
          className="text-4xl font-extrabold text-center text-green-900 mb-10"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {disease.replace(/_/g, " ")} – Disease & Cure
        </motion.h1>
        <div className="flex flex-col items-center">
          {/* Interactive container for image and info panel */}
          <motion.div
            className="flex w-full max-w-2xl overflow-hidden rounded-2xl shadow-2xl mb-8 cursor-pointer mx-auto"
            onClick={() => isMobile && setInfoVisible(!isInfoVisible)}
            initial="rest"
            whileHover={!isMobile ? "hover" : undefined}
            animate={isMobile ? (isInfoVisible ? "hover" : "rest") : "rest"}
          >
            <motion.div
              className="flex-shrink-0 flex items-center justify-center"
              variants={imageContainerVariants}
              transition={{ duration: 0.5 }}
            >
              <img
                src={cureData.carouselImage}
                alt={`${disease} info`}
                loading="lazy"
                className="object-cover w-full h-auto"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src =
                    "https://via.placeholder.com/600x400?text=Image+Not+Available";
                }}
              />
            </motion.div>
            <motion.div
              className="flex-grow bg-black bg-opacity-90 p-4 flex flex-col justify-center"
              variants={infoContainerVariants}
              transition={{ duration: 0.5 }}
            >
              <div className="flex flex-col sm:flex-row items-center gap-2 mb-2">
                <FaInfoCircle size={24} className="text-white" />
                <span className="text-white font-medium text-base sm:text-lg">
                  Disease Info
                </span>
                <FaChevronRight size={20} className="text-white" />
              </div>
              <p className="text-white whitespace-pre-line text-sm sm:text-base">
                {cureData.info}
              </p>
            </motion.div>
          </motion.div>
          {/* Cure Cards Slider */}
          <div className="w-full max-w-2xl mx-auto mt-4">
            <Slider {...sliderSettings}>
              {cureData.cards.map((card, idx) => (
                <div key={idx} className="p-3">
                  <motion.div
                    className="relative bg-white/80 backdrop-blur-sm rounded-xl shadow-lg p-6 transition duration-300"
                    initial={{ opacity: 0, y: 10 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    whileHover={{ scale: 1.1, zIndex: 10 }}
                    transition={{ duration: 0.5 }}
                  >
                    <motion.img
                      src={card.cardImage}
                      alt={card.title}
                      loading="lazy"
                      className="w-full h-40 object-cover rounded mb-4"
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.src =
                          "https://via.placeholder.com/150?text=No+Image";
                      }}
                      initial={{ opacity: 0 }}
                      whileInView={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    />
                    <motion.h3
                      className="font-semibold mb-4 text-center text-green-700 text-lg"
                      whileHover={{ scale: 1.05 }}
                    >
                      {card.title}
                    </motion.h3>
                    <motion.a
                      href={card.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-center bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded-full shadow transition duration-300"
                      initial={{ opacity: 0 }}
                      whileInView={{ opacity: 1 }}
                      whileHover={{ scale: 1.05 }}
                      transition={{ duration: 0.5 }}
                      aria-label={`Buy ${card.title}`}
                    >
                      Buy Now
                    </motion.a>
                    {/* Hover overlay for additional info (clicks pass through) */}
                    <motion.div
                      className="absolute inset-0 bg-black bg-opacity-75 flex items-center justify-center p-2 text-center pointer-events-none"
                      initial={{ opacity: 0 }}
                      whileHover={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      <p className="text-white text-sm">{card.hoverText}</p>
                    </motion.div>
                  </motion.div>
                </div>
              ))}
            </Slider>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CureLinks;
