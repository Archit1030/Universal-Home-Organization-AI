// script.js

document.addEventListener("DOMContentLoaded", function () {
  fetch("categorized_objects.json")
    .then((response) => response.json())
    .then((data) => {
      const categorizedObjects = {};

      // Step 1: Group objects by category
      data.objects.forEach((object) => {
        const category = object.category || "uncategorized";

        if (!categorizedObjects[category]) {
          categorizedObjects[category] = [];
        }

        categorizedObjects[category].push(object.name);
      });

      // Step 2: Display categories and items
      const container = document.getElementById("categories");
      Object.keys(categorizedObjects).forEach((category) => {
        const section = document.createElement("div");
        section.className = "category-section";

        const title = document.createElement("div");
        title.className = "category-title";
        title.textContent = category.charAt(0).toUpperCase() + category.slice(1);
        section.appendChild(title);

        const list = document.createElement("ul");
        categorizedObjects[category].forEach((objectName) => {
          const item = document.createElement("li");
          item.className = "object-item";
          item.textContent = objectName;
          list.appendChild(item);
        });

        section.appendChild(list);
        container.appendChild(section);
      });
    })
    .catch((error) => {
      console.error("Error loading JSON:", error);
    });
});
