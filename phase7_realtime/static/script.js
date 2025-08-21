// script.js
document.addEventListener('DOMContentLoaded', () => {
  const categoriesDiv = document.getElementById('categories');
  const rawDiv = document.getElementById('raw');

  async function fetchDetections() {
    try {
      const res = await fetch('/detections');
      if (!res.ok) throw new Error('Network response not ok');
      const data = await res.json();

      // show raw for debugging
      rawDiv.textContent = JSON.stringify(data, null, 2);

      // group by category
      const groups = {};
      if (data.objects && Array.isArray(data.objects)) {
        data.objects.forEach(obj => {
          const cat = obj.category || 'unknown';
          if (!groups[cat]) groups[cat] = [];
          groups[cat].push(obj);
        });
      }

      // render
      categoriesDiv.innerHTML = '';
      Object.keys(groups).forEach(cat => {
        const section = document.createElement('div');
        section.className = 'category-section';
        const title = document.createElement('h4');
        title.textContent = `${cat} (${groups[cat].length})`;
        section.appendChild(title);

        const ul = document.createElement('ul');
        groups[cat].forEach(item => {
          const li = document.createElement('li');
          li.textContent = `${item.name} — zone: ${item.zone} — conf: ${item.confidence}`;
          ul.appendChild(li);
        });

        section.appendChild(ul);
        categoriesDiv.appendChild(section);
      });

    } catch (err) {
      console.error('Error fetching detections:', err);
      categoriesDiv.textContent = 'Error fetching detections';
    }
  }

  // poll every 600ms
  setInterval(fetchDetections, 600);
  fetchDetections();
});
